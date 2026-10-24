"""Behavioral checks for visual control and causal ROS datasets (no ROS required)."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image
from agent.architecture.actions import ACTIONS, action_mask, validate_speed
from agent.architecture.capture import SessionWriter, prepare_session
from agent.architecture.policy import AgentPolicy
from agent.architecture.ros_sources import load_topics
from agent.architecture.targets import TARGET_DATASET_FORMAT, one_hot_action
from agent.architecture.train_targets import target_loss, augment_length_symmetry, read_target_datasets


def world(image=None):
    return {'sequence': 1, 'timestamp': 1., 'observer_monotonic_s': 10.,
            'robot': {'pose': {'x': 1., 'y': 1., 'yaw': 0.}, 'pose_age_s': 0., 'wrist_angle_rad': 0.},
            'sources': {'wrist': {'fresh': True, 'age_s': 0.}},
            'active_operation': None, 'last_action_result': None,
            'front_camera': {'path': str(image) if image else None, 'age_s': 0., 'valid': bool(image)}}


def sample(w, action='SPEED', group='one'):
    return {'format': TARGET_DATASET_FORMAT, 'source': 'test', 'group_id': group,
            'sample_id': f'{group}:{action}', 'observation': w, 'chosen_action': action,
            'target_action_probs': one_hot_action(action),
            'move_target_valid': action == 'MOVE',
            'target_move_pose': {'x': 1.1, 'y': 1.2, 'sin': 0., 'cos': 1.} if action == 'MOVE' else None,
            'speed_target_valid': action == 'SPEED',
            'target_speed': {'vx': .1, 'vy': -.02, 'omega': .2} if action == 'SPEED' else None}


class VisualControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.image = self.path / 'front.jpg'
        Image.new('RGB', (320, 240), (30, 60, 150)).save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def test_image_freshness_and_operation_masks(self):
        w = world(self.image)
        self.assertTrue(action_mask(w)[ACTIONS.index('SPEED')])
        for image in (None, {'path': str(self.image), 'age_s': .201}, {'path': str(self.image), 'age_s': -1}):
            w['front_camera'] = image
            self.assertFalse(action_mask(w)[ACTIONS.index('SPEED')])
            self.assertTrue(action_mask(w)[ACTIONS.index('MOVE')])
        w = world(self.image)
        w['active_operation'] = {'type': 'MOVE', 'status': 'running', 'cancelable': True}
        self.assertFalse(action_mask(w)[ACTIONS.index('SPEED')])
        self.assertTrue(action_mask(w)[ACTIONS.index('CANCEL')])
        self.assertNotIn('CENTRAGE', ACTIONS)

    def test_speed_direction_is_sin_cos_and_rotation_independent(self):
        policy = AgentPolicy()
        latent = torch.tensor([0., 1., 0., .3])
        speed = policy.decode_speed_tensor(latent)
        self.assertAlmostEqual(float(speed[0]), 0.)
        self.assertAlmostEqual(float(speed[1]), .125)
        self.assertAlmostEqual(float(speed[2]), math.tanh(.3), places=6)
        for _ in range(20):
            v = policy.decode_speed_tensor(torch.randn(4) * 10)
            self.assertLessEqual(float(torch.linalg.vector_norm(v[:2])), .250001)
            self.assertLessEqual(abs(float(v[2])), 1.000001)

    def test_supervised_speed_trains_camera_and_speed_not_pose_head(self):
        policy = AgentPolicy()
        loss, _ = target_loss(policy, sample(world(self.image)), torch.ones(len(ACTIONS)))
        loss.backward()
        self.assertGreater(float(policy.network.camera_encoder.cnn[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(policy.network.speed_head[-1].weight.grad.abs().sum()), 0.)
        self.assertIsNone(policy.network.move_mean_head[-1].weight.grad)
        policy.network.zero_grad()
        target_loss(policy, sample(world(self.image), 'MOVE'), torch.ones(len(ACTIONS)))[0].backward()
        self.assertIsNone(policy.network.speed_head[-1].weight.grad)

    def test_batched_images_and_checkpoint(self):
        policy = AgentPolicy()
        policy.network.eval()
        inputs = [world(self.image), world()]
        batch = policy.network.forward_batch(inputs)
        for w, output in zip(inputs, batch):
            expected = policy.network(w)
            torch.testing.assert_close(output['action_logits'], expected['action_logits'])
        path = self.path / 'policy.pt'
        policy.save_checkpoint(path)
        loaded, _ = AgentPolicy.from_checkpoint(path)
        torch.testing.assert_close(loaded.network(inputs[0])['action_logits'], batch[0]['action_logits'])
        torch.save({'format': 'not-a-policy'}, path)
        with self.assertRaisesRegex(ValueError, 'expected arcl-agent-policy'):
            loaded.load_checkpoint(path)

    def test_visual_samples_do_not_get_invalid_symmetry(self):
        rows = [sample(world(self.image)), sample(world(), 'MOVE')]
        self.assertEqual(len(augment_length_symmetry(rows)), 3)

    def test_forbidden_sources_and_invalid_images(self):
        for topic in ('/deported_camera/image', '/robot_omni/pose_vraie', '/rock_centering'):
            config = self.path / 'topics.json'
            config.write_text(json.dumps({'image': topic}))
            with self.assertRaises(ValueError):
                load_topics(config)
        self.image.write_bytes(b'corrupt jpeg')
        with self.assertRaisesRegex(ValueError, 'Cannot decode'):
            AgentPolicy().network(world(self.image))
        self.image.unlink()
        with self.assertRaisesRegex(ValueError, 'Cannot decode'):
            AgentPolicy().network(world(self.image))

    def test_causal_speed_samples_include_zero_and_survive_interruption(self):
        writer = SessionWriter(self.path / 'session', {'sample_hz': 10})
        for index in range(4):
            w = world(self.image)
            w.update(sequence=index, observer_monotonic_s=10. + index * .1)
            writer.write('observations', w, received=10. + index * .1)
        for index in range(15):
            at = 10.01 + index * .02
            event = {'format': 'arcl-command-event', 'id': str(index), 'command': 'speed',
                     'status': 'accepted', 'parameters': {'vx': .1 if index < 5 else 0., 'vy': 0., 'omega': .2}}
            writer.write('events', event, received=at)
        writer.close(complete=False)
        # Recover a truncated final event after an interrupted write.
        with (writer.path / 'events.jsonl').open('a') as stream:
            stream.write('{"unfinished"')
        output = self.path / 'targets.jsonl'
        self.assertEqual(prepare_session(writer.path, output), 3)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertTrue(all(r['chosen_action'] == 'SPEED' for r in rows))
        self.assertTrue(any(r['target_speed']['vx'] == 0 for r in rows))
        self.assertFalse(rows[0]['metadata']['session_complete'])
        self.assertEqual(rows[0]['observation']['sequence'], 0)
        self.assertEqual(rows[-1]['observation']['sequence'], 2)
        read_target_datasets([output])

    def test_final_stop_is_kept_inside_the_same_sampling_window(self):
        writer = SessionWriter(self.path / 'stop-session', {})
        writer.write('observations', world(self.image), received=10.)
        for at, velocity in ((10.01, .1), (10.08, 0.)):
            writer.write('events', {'format': 'arcl-command-event', 'id': str(at),
                'command': 'speed', 'status': 'accepted',
                'parameters': {'vx': velocity, 'vy': 0., 'omega': 0.}}, received=at)
        writer.close()
        output = self.path / 'stop-targets.jsonl'
        self.assertEqual(prepare_session(writer.path, output), 1)
        self.assertEqual(json.loads(output.read_text())['target_speed']['vx'], 0.)

    def test_future_image_is_excluded(self):
        writer = SessionWriter(self.path / 'future-session', {})
        observation = world(self.image)
        observation['front_camera']['timestamp'] = 2.
        writer.write('observations', observation, received=10.)
        writer.write('events', {'format': 'arcl-command-event', 'id': 'future',
            'command': 'speed', 'status': 'accepted',
            'parameters': {'vx': .1, 'vy': 0., 'omega': 0.}}, received=10.05)
        writer.close()
        output = self.path / 'future-targets.jsonl'
        self.assertEqual(prepare_session(writer.path, output), 0)
        self.assertEqual(json.loads(output.with_suffix('.preparation.json').read_text())['counts']['excluded:future image'], 1)

    def test_selfplay_relative_actions_have_no_fabricated_speed_supervision(self):
        from agent.rl.environment import _backend_action
        from agent.rl.prepare_targets import prepare_episode_targets
        observation = world(self.image)
        decision = {'action': 'SPEED', 'target_speed': {'vx': .1, 'vy': 0., 'omega': .2},
                    'behavior': {'action_probs': one_hot_action('SPEED')}}
        self.assertEqual(_backend_action(decision, observation)['duration_s'], .1)
        rows = [{'kind': 'transition', 'reward': {'a': 1., 'b': 0.},
                 'actions': {'a': decision, 'b': decision},
                 'agent_observations': {'a': observation, 'b': observation}},
                {'kind': 'episode_end', 'complete': True, 'rewards_complete': True,
                 'gamma': .99, 'robot_ids': ['a', 'b']}]
        targets = prepare_episode_targets('episode', rows)
        self.assertEqual(len(targets), 2)
        self.assertTrue(all(not row['speed_target_valid'] and row['target_speed'] is None for row in targets))

    def test_global_target_is_independent_of_command_client(self):
        targets = []
        for client in ('pupitre', 'strategie'):
            writer = SessionWriter(self.path / client, {})
            writer.write('observations', world(self.image), received=10.)
            writer.write('events', {'format': 'arcl-command-event', 'id': client,
                'command': 'move', 'status': 'accepted', 'client_hint': client,
                'parameters': {'x': 1.2, 'y': 1.3, 'theta': .2}}, received=10.05)
            writer.close()
            output = self.path / f'{client}.jsonl'
            prepare_session(writer.path, output)
            targets.append(json.loads(output.read_text()))
        self.assertEqual(targets[0]['chosen_action'], targets[1]['chosen_action'])
        self.assertEqual(targets[0]['target_move_pose'], targets[1]['target_move_pose'])

    def test_reject_nonfinite_or_over_limit_velocity(self):
        for speed in ({'vx': .2, 'vy': .2, 'omega': 0.}, {'vx': 0., 'vy': 0., 'omega': float('nan')}):
            with self.assertRaises(ValueError):
                validate_speed(world(self.image), speed)

if __name__ == '__main__':
    unittest.main()
