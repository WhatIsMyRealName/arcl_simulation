"""ROS integration smoke test; run explicitly after building and sourcing the workspace."""
import asyncio, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo, JointState
from rosgraph_msgs.msg import Clock
from robot_msgs.msg import RobotSpeed
from robot_msgs.action import GoToPose
from std_msgs.msg import String, Float64
from std_srvs.srv import SetBool
from agent.architecture.capture import prepare_session
from agent.architecture.action_executor import ActionExecutor
from agent.tests.test_visual_control import world
from PIL import Image as PILImage

root=Path(__file__).resolve().parents[2]; directory=Path(tempfile.mkdtemp(prefix='arcl-ros-validation-'))
os.environ['ROS_DOMAIN_ID'] = str(50 + os.getpid() % 100)
env={**os.environ,'ROS_LOG_DIR':str(directory/'logs')}
processes=[]; logs=[]
def launch(command):
    log=(directory/f'process-{len(processes)}.log').open('w');logs.append(log)
    p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);processes.append(p);return p
rclpy.init(); node=rclpy.create_node('visual_controller_validation')
posepub=node.create_publisher(PoseStamped,'/pose_estimation',10)
speedpub=node.create_publisher(RobotSpeed,'/speed_command',10)
imagepub=node.create_publisher(Image,'/camera/image',10)
infopub=node.create_publisher(CameraInfo,'/camera/camera_info',10)
clockpub=node.create_publisher(Clock,'/clock',10)
jointpub=node.create_publisher(JointState,'/joint_states',10)
events=[];wheels=[]
node.create_subscription(String,'/agent/command_events',lambda m:events.append(json.loads(m.data)),1000)
node.create_subscription(Float64,'/robot_omni/roue_1/cmd_vel',lambda m:wheels.append(m.data),100)
start=time.monotonic();last_image=0

def pump(seconds):
    global last_image
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        simulated=time.monotonic()-start
        clock=Clock();clock.clock.sec=int(simulated);clock.clock.nanosec=int((simulated%1)*1e9);clockpub.publish(clock)
        pose=PoseStamped();pose.header.frame_id='map';pose.header.stamp=clock.clock;pose.pose.position.x=1.;pose.pose.position.y=1.;pose.pose.orientation.w=1.;posepub.publish(pose)
        joints=JointState();joints.name=['poignet_joint'];joints.position=[0.];jointpub.publish(joints)
        if time.monotonic()-last_image>.05:
            image=Image();image.header.stamp=clock.clock;image.header.frame_id='camera_optical';image.width=160;image.height=120;image.encoding='rgb8';image.step=480;image.data=bytes([30,60,150])*(160*120);imagepub.publish(image)
            info=CameraInfo();info.header=image.header;info.width=160;info.height=120;info.k=[100.,0.,80.,0.,100.,60.,0.,0.,1.];infopub.publish(info)
            last_image=time.monotonic()
        rclpy.spin_once(node,timeout_sec=.01)

def wait(future,timeout=5):
    deadline=time.monotonic()+timeout
    while not future.done() and time.monotonic()<deadline:pump(.02)
    assert future.done(),'future timed out'
    return future.result()

def send_speed(vx=.1,vy=.03,omega=.1):
    message=RobotSpeed();message.vx=vx;message.vy=vy;message.omega=omega;speedpub.publish(message)

try:
    control=launch([str(root/'install/lib/arcl_simulation/robot_control'), '--ros-args', '-p', 'publish_command_events:=true'])
    services=launch([str(root/'install/lib/arcl_simulation/ventouse_service'), '--ros-args', '-p', 'publish_command_events:=true'])
    collector=launch([sys.executable,'-m','agent.architecture.collect_ros','--output',str(directory/'session')])
    pump(4)
    assert control.poll() is None and collector.poll() is None,'startup failed'
    for _ in range(35):send_speed();pump(.02)
    send_speed(0.,0.,0.);pump(.3)
    assert any(e['command']=='speed' and e['status']=='accepted' for e in events),'speed not accepted'
    send_speed();pump(.4)
    assert any(e['command']=='speed' and e['status']=='expired' for e in events),'watchdog did not expire'
    assert abs(wheels[-1])<1e-8,'watchdog did not stop wheels'
    client=ActionClient(node,GoToPose,'/go_to_pose');assert client.wait_for_server(timeout_sec=3)
    goal=GoToPose.Goal();goal.x=1.5;goal.y=1.;goal.theta=0.
    handle=wait(client.send_goal_async(goal));assert handle.accepted
    send_speed();pump(.2)
    assert events[-1]['command']=='speed' and events[-1]['status']=='rejected','speed not rejected during goto'
    wait(handle.cancel_goal_async());result=wait(handle.get_result_async());assert result.status==5
    send_speed();pump(.15);send_speed(0.,0.,0.);pump(.2)
    wrist=node.create_client(SetBool,'/robot_omni/poignet/set');assert wrist.wait_for_service(timeout_sec=3)
    request=SetBool.Request();request.data=True
    assert wait(wrist.call_async(request)).success
    assert any(e['command']=='wrist' and e['status']=='succeeded' for e in events)
    # Verify the learned-action executor publishes and stops a coupled translation/rotation.
    image_path = directory / 'executor-front.jpg'
    PILImage.new('RGB', (160, 120), (30, 60, 150)).save(image_path)
    executor = ActionExecutor()
    observation = world(image_path)
    executor.start()
    pump(.5)
    decision = {'action': 'SPEED', 'world_sequence': observation['sequence'],
                'target_speed': {'vx': .08, 'vy': .02, 'omega': .1}}
    response = asyncio.run(executor.dispatch(decision, observation))
    assert response['status'] == 'running'
    pump(.25)
    assert abs(wheels[-1]) < 1e-8, 'executor did not stop after 100 ms'
    asyncio.run(executor.close())
    collector.send_signal(signal.SIGINT);collector.wait(timeout=10)
    count=prepare_session(directory/'session',directory/'targets.jsonl')
    rows=[json.loads(line) for line in (directory/'targets.jsonl').read_text().splitlines()]
    actions={r['chosen_action'] for r in rows}
    assert 'SPEED' in actions and 'MOVE' in actions and 'TOGGLE_WRIST' in actions,(actions,count)
    assert 'CENTRAGE' not in actions
    assert any(r['observation']['front_camera'].get('path') for r in rows)
    print(json.dumps({'directory':str(directory),'events':len(events),'targets':count,'actions':sorted(actions),'watchdog':True,'mutual_exclusion':True,'service_events':True}))
finally:
    for p in reversed(processes):
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try:p.wait(timeout=7)
            except subprocess.TimeoutExpired:p.kill();p.wait()
    for log in logs:log.close()
    node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
