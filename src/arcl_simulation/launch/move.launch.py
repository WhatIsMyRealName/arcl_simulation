#!/usr/bin/env python3
"""Pupitre du robot_omni : ventouses, ascenseur, poignet et base mobile.

Services (nœud ventouse_service) :
  /robot_omni/ventouse/set       les 3 étages ensemble
  /robot_omni/ventouse_<n>/set   un étage (1 = carton du bas quand le poignet est à 0°)
  /robot_omni/poignet/set        true = 90° (cartons debout), false = 0°
Actions (nœud robot_control) :
  /go_to_pose       aller en (x, y, theta)  — feedback : elapsed, message
                    (cible choisie à la souris sur la carte, comme « 2D Goal Pose » dans RViz)
  /rock_centering   centrage sur le marqueur — feedback : marker_visible, err_fwd, err_lat, err_theta, elapsed
Affiche aussi la position de l'ascenseur et du poignet lue sur /joint_states.

Lancer :  python3 interface_ventouses.py    (après avoir sourcé install/setup.bash)
Prérequis : sudo apt install python3-tk
"""

import math
import json
import os
import queue
import subprocess
import sys
import threading
import time
import copy
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.action.graph import get_action_names_and_types
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_action
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool


def _workspace_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / 'agent' / 'architecture' / 'world_observer.py').is_file():
            return candidate
    candidate = Path.cwd().resolve()
    if (candidate / 'agent' / 'architecture' / 'world_observer.py').is_file():
        return candidate
    return None


WORKSPACE_ROOT = _workspace_root()
if WORKSPACE_ROOT is not None and str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
try:
    from agent.architecture.demonstrations import DemonstrationRecorder, operator_choice_to_decision
    from agent.architecture.actions import ACTIONS as POLICY_ACTIONS
    from agent.architecture.targets import DEFAULT_TARGET_DATASET_PATH
except ImportError:
    DemonstrationRecorder = None
    operator_choice_to_decision = None
    POLICY_ACTIONS = ()
    DEFAULT_TARGET_DATASET_PATH = Path('runs/targets.jsonl')

try:
    from tf2_ros import Buffer, TransformListener
except ImportError:      # sans tf2_ros, la carte marche mais n'affiche pas le robot
    Buffer = TransformListener = None

SERVICES = {
    'tous': '/robot_omni/ventouse/set',
    'etage_1': '/robot_omni/ventouse_1/set',
    'etage_2': '/robot_omni/ventouse_2/set',
    'etage_3': '/robot_omni/ventouse_3/set',
    'poignet': '/robot_omni/poignet/set',
}

# 'type' = None : détecté automatiquement sur le graphe ROS (comme `ros2 action info -t`).
# On peut le forcer, par exemple 'mon_paquet/action/GoToPose'.
ACTIONS = {
    'go_to_pose': {'nom': '/go_to_pose', 'type': None},
    'rock_centering': {'nom': '/rock_centering', 'type': None},
}

ECHELLE_CIBLE = 0.10   # m : erreur affichée au bord de la cible de centrage

# Carte cliquable de /go_to_pose. Les bornes sont celles du repère dans lequel le serveur
# attend x, y, theta : à adapter à ton arène.
CARTE = {
    'x_min': 0.0, 'x_max': 3.0,
    'y_min': 0.0, 'y_max': 2.0,
    'marge_robot': 0.20,
    'pas': 0.5,              # m entre deux lignes de grille
    'repere': 'map',         # frame des poses, pour afficher le robot via TF
    'robot': 'base_link',
    'taille_robot': 0.30,    # m, côté du carré dessiné
}
PETITE = ('TkDefaultFont', 8)

C = {
    'fond': '#f4f5f7', 'carte': '#ffffff', 'texte': '#1f2328', 'doux': '#6b7280', 'table': '#fbfaf5',
    'accent': '#2563eb', 'accent_actif': '#1d4ed8', 'accent_off': '#a9bff0',
    'danger': '#dc2626', 'danger_actif': '#b91c1c', 'danger_off': '#efb4b4',
    'ok': '#16a34a', 'erreur': '#b3261e', 'bordure': '#d0d4da', 'grille': '#e5e7eb',
}


class Pont(Node):
    """Côté ROS : clients de service, clients d'action et abonnement à /joint_states.

    Tourne dans un thread à part ; tout ce qui doit toucher l'interface passe par la file `sortie`.
    """

    def __init__(self, sortie):
        super().__init__('interface_ventouses')
        self.sortie = sortie
        self.clients_srv = {k: self.create_client(SetBool, v) for k, v in SERVICES.items()}
        self.create_subscription(JointState, '/joint_states', self.sur_joints, 10)

        self.actions = {}            # cle -> ActionClient
        self.types_action = {}       # cle -> classe d'action
        self.goals = {}              # cle -> goal handle en cours
        self.annulation_demandee = set()
        self.types_signales = set()
        self.create_timer(1.0, self.chercher_actions)

        if Buffer is not None:
            self.tf_buffer = Buffer()
            self.tf_ecoute = TransformListener(self.tf_buffer, self)
            self.create_timer(0.1, self.lire_pose)

    # --- services ---
    def appeler(self, cle, valeur, libelle):
        client = self.clients_srv[cle]
        if not client.service_is_ready():
            self.sortie.put(('fin', cle, 'failed', f'{libelle} : service {SERVICES[cle]} indisponible'))
            return
        requete = SetBool.Request()
        requete.data = valeur
        debut = time.monotonic()
        futur = client.call_async(requete)

        def termine(f):
            duree = time.monotonic() - debut
            try:
                rep = f.result()
                status = 'succeeded' if rep.success else 'failed'
                self.sortie.put(('fin', cle, status, f'{libelle} : {rep.message} ({duree:.1f} s)'))
            except Exception as err:  # noqa: BLE001 - toute erreur est affichée telle quelle
                self.sortie.put(('fin', cle, 'failed', f'{libelle} : erreur {err}'))

        futur.add_done_callback(termine)

    # --- actions ---
    def chercher_actions(self):
        """Crée les clients d'action dès que leur type est connu (appelé par un timer)."""
        manquantes = [k for k in ACTIONS if k not in self.actions]
        if not manquantes:
            return
        connues = dict(get_action_names_and_types(self))
        for cle in manquantes:
            conf = ACTIONS[cle]
            type_str = conf['type'] or (connues.get(conf['nom']) or [None])[0]
            if not type_str:
                continue
            try:
                classe = get_action(type_str)
            except Exception as err:  # noqa: BLE001
                if type_str not in self.types_signales:
                    self.types_signales.add(type_str)
                    self.sortie.put(('info', f'Type {type_str} non importable ({err}). '
                                             'As-tu sourcé le workspace qui le définit ?'))
                continue
            self.types_action[cle] = classe
            self.actions[cle] = ActionClient(self, classe, conf['nom'])
            self.sortie.put(('info', f'Action {conf["nom"]} : type {type_str}'))

    def action_prete(self, cle):
        client = self.actions.get(cle)
        return client is not None and client.server_is_ready()

    def envoyer_action(self, cle, champs, libelle):
        nom = ACTIONS[cle]['nom']
        if not self.action_prete(cle):
            self.sortie.put(('action_fin', cle, 'failed', f'{libelle} : serveur {nom} indisponible'))
            return
        but = self.types_action[cle].Goal()
        for champ, valeur in champs.items():
            setattr(but, champ, valeur)
        self.annulation_demandee.discard(cle)
        debut = time.monotonic()
        futur = self.actions[cle].send_goal_async(
            but, feedback_callback=lambda fb: self.sortie.put(('fb', cle, fb.feedback)))

        def accepte(f):
            try:
                gh = f.result()
            except Exception as err:  # noqa: BLE001
                self.sortie.put(('action_fin', cle, 'failed', f'{libelle} : erreur {err}'))
                return
            if not gh.accepted:
                self.sortie.put(('action_fin', cle, 'rejected', f'{libelle} : refusé par le serveur'))
                return
            self.goals[cle] = gh
            self.sortie.put(('action_started', cle))
            self.sortie.put(('info', f'{libelle} : accepté'))
            if cle in self.annulation_demandee:
                gh.cancel_goal_async()
            gh.get_result_async().add_done_callback(lambda r: self.resultat(cle, libelle, debut, r))

        futur.add_done_callback(accepte)

    def resultat(self, cle, libelle, debut, futur):
        self.goals.pop(cle, None)
        duree = time.monotonic() - debut
        try:
            res = futur.result()
        except Exception as err:  # noqa: BLE001
            self.sortie.put(('action_fin', cle, 'failed', f'{libelle} : erreur {err}'))
            return
        rep = res.result
        if res.status == GoalStatus.STATUS_SUCCEEDED:
            status = 'succeeded' if rep.success else 'failed'
            texte = rep.message or ('réussi' if rep.success else 'échec')
        elif res.status == GoalStatus.STATUS_CANCELED:
            status, texte = 'canceled', 'annulé' + (f' — {rep.message}' if rep.message else '')
        else:
            status, texte = 'failed', 'abandonné' + (f' — {rep.message}' if rep.message else '')
        self.sortie.put(('action_fin', cle, status, f'{libelle} : {texte} ({duree:.1f} s)'))

    def annuler(self, cle):
        self.annulation_demandee.add(cle)
        gh = self.goals.get(cle)
        if gh is not None:
            gh.cancel_goal_async()

    def annuler_tout(self):
        en_cours = list(self.goals)
        for cle in en_cours:
            self.annuler(cle)
        return bool(en_cours)

    # --- état ---
    def lire_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(CARTE['repere'], CARTE['robot'], Time())
        except Exception:  # noqa: BLE001 - TF pas encore disponible : on réessaie au prochain tour
            return
        t, q = tf.transform.translation, tf.transform.rotation
        lacet = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.sortie.put(('pose', t.x, t.y, lacet))

    def disponibilites(self):
        dispo = {SERVICES[k]: c.service_is_ready() for k, c in self.clients_srv.items()}
        dispo.update({ACTIONS[k]['nom']: self.action_prete(k) for k in ACTIONS})
        return dispo

    def sur_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        self.sortie.put(('joints', pos.get('ascenseur_joint'), pos.get('poignet_joint')))


class FluxObservations:
    """Run the canonical agent observer and forward its JSON snapshots to Tk."""

    def __init__(self, sortie, workspace_root):
        self.sortie = sortie
        self.workspace_root = workspace_root
        self.process = None
        self.threads = []
        self.stopping = threading.Event()

    def start(self):
        if self.process is not None:
            return
        if self.workspace_root is None:
            raise RuntimeError('racine du dépôt contenant agent/ introuvable')
        environment = os.environ.copy()
        current_path = environment.get('PYTHONPATH', '')
        environment['PYTHONPATH'] = str(self.workspace_root) + (
            os.pathsep + current_path if current_path else ''
        )
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'agent.architecture.world_observer',
             '--carton-source', 'gazebo_sim', '--publish-hz', '10'],
            cwd=self.workspace_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        self.threads = [
            threading.Thread(target=self._read_snapshots, daemon=True),
            threading.Thread(target=self._read_errors, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _read_snapshots(self):
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            try:
                observation = json.loads(line)
            except json.JSONDecodeError:
                self.sortie.put(('observer_error', 'snapshot JSON invalide'))
                continue
            if isinstance(observation, dict):
                self.sortie.put(('snapshot', observation))
        if not self.stopping.is_set():
            code = self.process.poll()
            self.sortie.put(('observer_stopped', code))

    def _read_errors(self):
        if self.process is None or self.process.stderr is None:
            return
        for line in self.process.stderr:
            message = line.strip()
            if message:
                self.sortie.put(('observer_info', message))

    def stop(self):
        if self.process is None:
            return
        self.stopping.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        for thread in self.threads:
            thread.join(timeout=1.0)
        self.process = None
        self.threads = []


def appliquer_style(racine):
    s = ttk.Style(racine)
    if 'clam' in s.theme_names():
        s.theme_use('clam')
    racine.configure(background=C['fond'])
    s.configure('.', background=C['fond'], foreground=C['texte'])
    s.configure('TLabelframe', background=C['fond'], bordercolor=C['bordure'])
    s.configure('TLabelframe.Label', background=C['fond'], foreground=C['texte'],
                font=('TkDefaultFont', 10, 'bold'))
    s.configure('Titre.TLabel', font=('TkDefaultFont', 14, 'bold'))
    s.configure('Valeur.TLabel', font=('TkFixedFont', 12))
    s.configure('Doux.TLabel', foreground=C['doux'])
    s.configure('TButton', padding=(10, 5))
    s.configure('Accent.TButton', background=C['accent'], foreground='white', bordercolor=C['accent'])
    s.map('Accent.TButton',
          background=[('disabled', C['accent_off']), ('active', C['accent_actif'])],
          foreground=[('disabled', '#f1f5fd')])
    s.configure('Danger.TButton', background=C['danger'], foreground='white', bordercolor=C['danger'])
    s.map('Danger.TButton',
          background=[('disabled', C['danger_off']), ('active', C['danger_actif'])],
          foreground=[('disabled', '#fdf3f3')])
    s.configure('Horizontal.TProgressbar', background=C['accent'], troughcolor=C['grille'],
                bordercolor=C['bordure'])


class Interface:
    def __init__(self, racine, pont, file_ros):
        self.racine = racine
        self.pont = pont
        self.file = file_ros
        self.en_cours = None          # une seule commande de service à la fois, comme le nœud
        self.boutons = []
        self.action_en_cours = None   # une seule action de base à la fois (les deux bougent la base)
        self.but_en_attente = None    # nouvelle cible donnée pendant un déplacement
        self.pose_robot = None        # (x, y, lacet) lu sur TF
        self.trace = []
        self.cible_pose = None        # dernière cible envoyée
        self.saisie = None            # cible en cours de saisie à la souris
        self.appui_px = (0, 0)
        self.observation_stream = None
        self.latest_observation = None
        self.latest_observation_at = None
        self.recording_starting = False
        self.recorder = None
        self.decision_recorders = {}
        self.operation_decision_id = None
        self.cancel_decision_id = None
        self.demo_active_operation = None
        self.demo_last_result = None
        self.demo_idle_timer = None

        racine.title('Pupitre robot_omni')
        racine.minsize(1000, 0)
        appliquer_style(racine)

        cadre = ttk.Frame(racine, padding=14)
        cadre.grid(sticky='nsew')
        racine.columnconfigure(0, weight=1)
        racine.rowconfigure(0, weight=1)
        cadre.columnconfigure(0, weight=2)
        cadre.columnconfigure(1, weight=3)
        cadre.rowconfigure(1, weight=3)

        bandeau = ttk.Frame(cadre)
        bandeau.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        bandeau.columnconfigure(0, weight=1)
        ttk.Label(bandeau, text='Pupitre robot_omni', style='Titre.TLabel').grid(
            row=0, column=0, sticky='w')
        self.btn_start_recording = ttk.Button(
            bandeau, text='Enregistrer une démonstration', command=self.start_recording
        )
        self.btn_start_recording.grid(row=0, column=1, padx=(8, 4))
        self.btn_stop_recording = ttk.Button(
            bandeau, text='Arrêter', command=self.stop_recording, state='disabled'
        )
        self.btn_stop_recording.grid(row=0, column=2, padx=(0, 8))
        self.lbl_recording = ttk.Label(bandeau, text='Enregistrement désactivé', style='Doux.TLabel')
        self.lbl_recording.grid(row=0, column=3, sticky='e')

        gauche = ttk.Frame(cadre)
        gauche.grid(row=1, column=0, sticky='nsew', padx=(0, 8))
        droite = ttk.Frame(cadre)
        droite.grid(row=1, column=1, sticky='nsew', padx=(8, 0))
        self.construire_prehension(gauche)
        self.construire_base(droite)

        # --- état des connexions et journal ---
        self.lbl_srv = ttk.Label(cadre, text='Recherche des services et actions…',
                                 style='Doux.TLabel', wraplength=850)
        self.lbl_srv.grid(row=2, column=0, columnspan=2, sticky='w', pady=(12, 0))
        bloc = ttk.Frame(cadre)
        bloc.grid(row=3, column=0, columnspan=2, sticky='nsew', pady=(4, 0))
        cadre.rowconfigure(3, weight=1)
        bloc.columnconfigure(0, weight=1)
        bloc.rowconfigure(0, weight=1)
        self.journal = tk.Text(bloc, height=6, state='disabled', wrap='word', relief='flat',
                               background=C['carte'], foreground=C['texte'], padx=8, pady=6,
                               highlightthickness=1, highlightbackground=C['bordure'],
                               font=('TkFixedFont', 10))
        self.journal.grid(row=0, column=0, sticky='nsew')
        defil = ttk.Scrollbar(bloc, orient='vertical', command=self.journal.yview)
        defil.grid(row=0, column=1, sticky='ns')
        self.journal.configure(yscrollcommand=defil.set)
        self.journal.tag_configure('ok', foreground='#1d7a3e')
        self.journal.tag_configure('erreur', foreground=C['erreur'])
        self.journal.tag_configure('info', foreground=C['doux'])

        racine.protocol('WM_DELETE_WINDOW', self.fermer)
        self.racine.after(50, self.lire_file)
        self.racine.after(500, self.verifier_services)

    def start_recording(self):
        if DemonstrationRecorder is None or operator_choice_to_decision is None:
            self.ecrire('Enregistrement indisponible : module agent introuvable', 'erreur')
            return
        if self.recording_starting or self.recorder is not None:
            return
        try:
            self.observation_stream = FluxObservations(self.file, WORKSPACE_ROOT)
            self.observation_stream.start()
        except (OSError, RuntimeError) as error:
            self.observation_stream = None
            self.ecrire(f'Impossible de démarrer l’observateur : {error}', 'erreur')
            return
        self.recording_starting = True
        self.btn_start_recording.state(['disabled'])
        self.lbl_recording.configure(text='Attente du premier snapshot ROS…', foreground=C['doux'])

    def stop_recording(self, reason='operator stopped recording'):
        self.recording_starting = False
        if self.recorder is not None:
            recorder = self.recorder
            recorder.stop(reason=reason)
            if recorder.pending_results:
                self.ecrire(
                    'Enregistrement arrêté ; le fichier sera publié après la fin de l’ordre actif.',
                    'info',
                )
            elif recorder.published_samples:
                self.ecrire(
                    f'{recorder.published_samples} exemple(s) ajouté(s) à {recorder.path}', 'info'
                )
            else:
                self.ecrire('Aucun exemple à exporter.', 'info')
            self.recorder = None
        if self.demo_idle_timer is not None:
            self.racine.after_cancel(self.demo_idle_timer)
            self.demo_idle_timer = None
        if self.observation_stream is not None:
            self.observation_stream.stop()
            self.observation_stream = None
        self.btn_start_recording.state(['!disabled'])
        self.btn_stop_recording.state(['disabled'])
        self.lbl_recording.configure(text='Enregistrement désactivé', foreground=C['doux'])

    def _activate_recorder(self):
        if not self.recording_starting or self.latest_observation is None:
            return
        destination = WORKSPACE_ROOT / DEFAULT_TARGET_DATASET_PATH
        try:
            self.recorder = DemonstrationRecorder(destination)
        except OSError as error:
            self.recording_starting = False
            if self.observation_stream is not None:
                self.observation_stream.stop()
                self.observation_stream = None
            self.btn_start_recording.state(['!disabled'])
            self.lbl_recording.configure(text='Échec de création du fichier', foreground=C['erreur'])
            self.ecrire(f'Impossible de créer la session de démonstration : {error}', 'erreur')
            return
        self.recording_starting = False
        self.btn_stop_recording.state(['!disabled'])
        self.lbl_recording.configure(
            text=f'Enregistrement : {self.recorder.session_id[:8]}', foreground=C['ok']
        )
        self.ecrire(f'Session de démonstration démarrée : {self.recorder.path}', 'ok')
        self._schedule_demo_idle()

    def _observation_for_decision(self):
        if self.latest_observation is None:
            return None
        observation = copy.deepcopy(self.latest_observation)
        observation['active_operation'] = copy.deepcopy(self.demo_active_operation)
        observation['last_action_result'] = copy.deepcopy(self.demo_last_result)
        return observation

    def _record_choice(self, command, parameters, *, requires_result=True):
        if self.recorder is None:
            return None, None
        observation = self._observation_for_decision()
        if observation is None:
            self.ecrire('Choix non exporté : aucun snapshot agent disponible', 'erreur')
            return None, None
        decision = operator_choice_to_decision(command, parameters, observation)
        if decision.get('action') not in POLICY_ACTIONS:
            return None, decision
        decision['world_sequence'] = int(observation.get('sequence', -1))
        decision['world_timestamp'] = observation.get('timestamp')
        decision_id = self.recorder.record_decision(
            observation, decision, command=command, parameters=parameters,
            requires_result=requires_result,
        )
        if requires_result:
            self.decision_recorders[decision_id] = self.recorder
        return decision_id, decision

    def _operator_decision(self, command, parameters):
        if operator_choice_to_decision is None:
            return {'action': 'UNSUPPORTED'}
        observation = self._observation_for_decision() or {'robot': {}}
        return operator_choice_to_decision(command, parameters, observation)

    def _record_result(self, decision_id, action, status, reason):
        recorder = self.decision_recorders.pop(decision_id, None) if decision_id is not None else None
        if recorder is not None:
            recorded = recorder.record_result(decision_id, {
                'action': action, 'status': status, 'failed': status in ('failed', 'rejected'),
                'reason': reason,
            })
            if recorded and recorder.path.exists():
                self.ecrire(f'Fichier d’entraînement enregistré : {recorder.path}', 'info')
        self.demo_last_result = {
            'id': decision_id, 'type': action, 'status': status,
            'failed': status in ('failed', 'rejected'), 'cancelable': False,
            'target_pose': copy.deepcopy(
                (self.demo_active_operation or {}).get('target_pose')
            ),
        }

    def _schedule_demo_idle(self):
        if self.recorder is not None and self.demo_idle_timer is None:
            self.demo_idle_timer = self.racine.after(100, self._record_implicit_idle)

    def _record_implicit_idle(self):
        self.demo_idle_timer = None
        if self.recorder is None:
            return
        if (self.demo_active_operation is not None
                and self.demo_active_operation.get('type') != 'UNSUPPORTED_ROCK_CENTERING'):
            self._record_choice('idle', {}, requires_result=False)
        self._schedule_demo_idle()

    def _receive_observation(self, observation):
        self.latest_observation = observation
        self.latest_observation_at = time.monotonic()
        if self.pose_robot is None:
            robot = observation.get('robot') if isinstance(observation.get('robot'), dict) else {}
            pose = robot.get('pose') if isinstance(robot.get('pose'), dict) else None
            if pose is not None:
                self.pose_robot = (pose.get('x'), pose.get('y'), pose.get('yaw'))
        if self.recording_starting:
            self._activate_recorder()

    # ------------------------------------------------------------------ construction
    def construire_prehension(self, parent):
        parent.columnconfigure(0, weight=1)
        ligne = 0

        etat = ttk.LabelFrame(parent, text='État (joint_states)', padding=10)
        etat.grid(row=ligne, column=0, sticky='ew'); ligne += 1
        ttk.Label(etat, text='Ascenseur').grid(row=0, column=0, sticky='w')
        self.lbl_asc = ttk.Label(etat, text='—', style='Valeur.TLabel', width=10, anchor='e')
        self.lbl_asc.grid(row=0, column=1, sticky='e')
        ttk.Label(etat, text='Poignet').grid(row=1, column=0, sticky='w')
        self.lbl_poi = ttk.Label(etat, text='—', style='Valeur.TLabel', width=10, anchor='e')
        self.lbl_poi.grid(row=1, column=1, sticky='e')
        etat.columnconfigure(0, weight=1)

        tous = ttk.LabelFrame(parent, text='Les 3 étages ensemble', padding=10)
        tous.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        self.bouton(tous, 'Prendre la pile', 'tous', True, 'Prendre les 3', 'Accent.TButton').grid(
            row=0, column=0, sticky='ew', padx=(0, 4))
        self.bouton(tous, 'Tout lâcher', 'tous', False, 'Lâcher les 3').grid(
            row=0, column=1, sticky='ew', padx=(4, 0))
        tous.columnconfigure((0, 1), weight=1)

        # étages affichés du haut vers le bas, comme sur le robot à 0°
        etages = ttk.LabelFrame(parent, text='Étage par étage', padding=10)
        etages.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        for i, n in enumerate((3, 2, 1)):
            nom = {3: 'Étage 3 (haut)', 2: 'Étage 2 (milieu)', 1: 'Étage 1 (bas)'}[n]
            ttk.Label(etages, text=nom).grid(row=i, column=0, sticky='w', pady=2)
            cle = f'etage_{n}'
            self.bouton(etages, 'Prendre', cle, True, f'Prendre étage {n}').grid(row=i, column=1, padx=4, pady=2)
            self.bouton(etages, 'Poser', cle, False, f'Poser étage {n}').grid(row=i, column=2, pady=2)
        etages.columnconfigure(0, weight=1)
        ttk.Label(etages, text='Poser : l\'ascenseur descend, la ventouse lâche, puis il remonte\n'
                               'si d\'autres étages tiennent encore un carton.',
                  style='Doux.TLabel').grid(row=3, column=0, columnspan=3, sticky='w', pady=(6, 0))

        poignet = ttk.LabelFrame(parent, text='Poignet', padding=10)
        poignet.grid(row=ligne, column=0, sticky='ew', pady=(10, 0)); ligne += 1
        self.bouton(poignet, 'Couché (0°)', 'poignet', False, 'Poignet à 0°').grid(
            row=0, column=0, sticky='ew', padx=(0, 4))
        self.bouton(poignet, 'Debout (90°)', 'poignet', True, 'Poignet à 90°').grid(
            row=0, column=1, sticky='ew', padx=(4, 0))
        poignet.columnconfigure((0, 1), weight=1)

    def construire_base(self, parent):
        parent.columnconfigure(0, weight=1)
        self.annulers = {}
        self.progres = {}

        # --- aller à une pose : carte cliquable, comme « 2D Goal Pose » de RViz ---
        nav = ttk.LabelFrame(parent, text='Aller à une pose   /go_to_pose', padding=10)
        nav.grid(row=0, column=0, sticky='nsew')
        parent.rowconfigure(0, weight=1)
        nav.columnconfigure(0, weight=1)
        nav.rowconfigure(1, weight=1)
        ttk.Label(nav, text='Clic gauche : position   ·   glisser : orientation   ·   '
                            'relâcher : envoyer   ·   clic droit : annuler',
                  style='Doux.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 6))
        self.carte = tk.Canvas(nav, height=300, background=C['carte'], cursor='crosshair',
                               highlightthickness=1, highlightbackground=C['bordure'])
        self.carte.grid(row=1, column=0, sticky='nsew')
        self.carte.bind('<Configure>', lambda _e: self.dessiner_carte())
        self.carte.bind('<ButtonPress-1>', self.carte_appui)
        self.carte.bind('<B1-Motion>', self.carte_glisse)
        self.carte.bind('<ButtonRelease-1>', self.carte_relache)
        self.carte.bind('<ButtonPress-3>', lambda _e: self.annuler('go_to_pose'))
        self.carte.bind('<Motion>', self.carte_survol)
        self.carte.bind('<Leave>', lambda _e: self.lbl_curseur.configure(text=''))

        sous = ttk.Frame(nav)
        sous.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        sous.columnconfigure(0, weight=1)
        self.lbl_cible = ttk.Label(sous, text='Cible : —')
        self.lbl_cible.grid(row=0, column=0, sticky='w')
        self.lbl_curseur = ttk.Label(sous, text='', style='Doux.TLabel')
        self.lbl_curseur.grid(row=0, column=1, sticky='e')

        boutons = ttk.Frame(nav)
        boutons.grid(row=3, column=0, sticky='ew', pady=(8, 0))
        boutons.columnconfigure((0, 1), weight=1)
        self.btn_aller = ttk.Button(boutons, text='Renvoyer la cible', style='Accent.TButton',
                                    command=self.envoyer_cible)
        self.btn_aller.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self.btn_aller.state(['disabled'])
        self.annulers['go_to_pose'] = ttk.Button(boutons, text='Annuler', style='Danger.TButton',
                                                 command=lambda: self.annuler('go_to_pose'))
        self.annulers['go_to_pose'].grid(row=0, column=1, sticky='ew', padx=(4, 0))

        self.progres['go_to_pose'] = ttk.Progressbar(nav, mode='indeterminate')
        self.progres['go_to_pose'].grid(row=4, column=0, sticky='ew', pady=(8, 4))
        info = ttk.Frame(nav)
        info.grid(row=5, column=0, sticky='ew')
        info.columnconfigure(0, weight=1)
        self.lbl_nav_msg = ttk.Label(info, text='En attente', style='Doux.TLabel', wraplength=420)
        self.lbl_nav_msg.grid(row=0, column=0, sticky='w')
        self.lbl_nav_t = ttk.Label(info, text='', style='Valeur.TLabel')
        self.lbl_nav_t.grid(row=0, column=1, sticky='e')

        # --- centrage sur le rocher ---
        cen = ttk.LabelFrame(parent, text='Centrage sur le rocher   /rock_centering', padding=10)
        cen.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        cen.columnconfigure(1, weight=1)
        self.cible = tk.Canvas(cen, width=170, height=170, background=C['carte'],
                               highlightthickness=1, highlightbackground=C['bordure'])
        self.cible.grid(row=0, column=0, rowspan=6, padx=(0, 12), sticky='n')

        self.lbl_marqueur = ttk.Label(cen, text='●  Marqueur : —', foreground=C['doux'],
                                      font=('TkDefaultFont', 10, 'bold'))
        self.lbl_marqueur.grid(row=0, column=1, columnspan=2, sticky='w', pady=(0, 6))
        self.lbl_cen = {}
        mesures = (('Erreur avant', 'err_fwd'), ('Erreur latérale', 'err_lat'),
                   ('Erreur angle', 'err_theta'), ('Durée', 'elapsed'))
        for i, (texte, champ) in enumerate(mesures):
            ttk.Label(cen, text=texte).grid(row=1 + i, column=1, sticky='w')
            lbl = ttk.Label(cen, text='—', style='Valeur.TLabel', width=10, anchor='e')
            lbl.grid(row=1 + i, column=2, sticky='e')
            self.lbl_cen[champ] = lbl

        bcen = ttk.Frame(cen)
        bcen.grid(row=5, column=1, columnspan=2, sticky='sew', pady=(8, 0))
        bcen.columnconfigure((0, 1), weight=1)
        self.btn_centrer = ttk.Button(bcen, text='Centrer', style='Accent.TButton',
                                      command=lambda: self.lancer_action('rock_centering', {}, 'Centrage rocher'))
        self.btn_centrer.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self.annulers['rock_centering'] = ttk.Button(bcen, text='Annuler', style='Danger.TButton',
                                                     command=lambda: self.annuler('rock_centering'))
        self.annulers['rock_centering'].grid(row=0, column=1, sticky='ew', padx=(4, 0))

        self.progres['rock_centering'] = ttk.Progressbar(cen, mode='indeterminate')
        self.progres['rock_centering'].grid(row=6, column=0, columnspan=3, sticky='ew', pady=(10, 0))

        for b in self.annulers.values():
            b.state(['disabled'])
        self.dessiner_cible()

    def bouton(self, parent, texte, cle, valeur, libelle, style='TButton'):
        b = ttk.Button(parent, text=texte, style=style, command=lambda: self.envoyer(cle, valeur, libelle))
        self.boutons.append(b)
        return b

    # ------------------------------------------------------------------ cible de centrage
    def dessiner_cible(self, fwd=None, lat=None, theta=None, visible=False):
        c = self.cible
        c.delete('all')
        cx = cy = 85
        r = 70
        for rr in (r, r / 2):
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=C['grille'])
        c.create_line(cx - r, cy, cx + r, cy, fill=C['grille'])
        c.create_line(cx, cy - r, cx, cy + r, fill=C['grille'])
        c.create_text(cx, 7, text='avant', fill=C['doux'], font=('TkDefaultFont', 8))
        c.create_text(14, cy - 8, text='gauche', fill=C['doux'], font=('TkDefaultFont', 8), anchor='w')
        c.create_text(166, 164, text=f'±{ECHELLE_CIBLE * 100:.0f} cm', fill=C['doux'],
                      font=('TkDefaultFont', 8), anchor='se')
        if fwd is None:
            return
        k = r / ECHELLE_CIBLE
        dx, dy = -lat * k, -fwd * k          # avant vers le haut, gauche vers la gauche
        d = math.hypot(dx, dy)
        hors_champ = d > r
        if hors_champ:
            dx, dy = dx * r / d, dy * r / d
        x, y = cx + dx, cy + dy
        couleur = C['ok'] if visible else C['doux']
        c.create_line(cx, cy, x, y, fill=couleur, dash=(2, 3))
        c.create_line(x, y, x - math.sin(theta) * 22, y - math.cos(theta) * 22,
                      fill=couleur, width=3, arrow='last')
        c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=couleur if not hors_champ else C['carte'],
                      outline=couleur, width=2)

    # ------------------------------------------------------------------ commandes
    def envoyer(self, cle, valeur, libelle):
        if self.en_cours or self.action_en_cours:
            self.ecrire('Attends la fin de l’opération en cours avant de commander la préhension', 'erreur')
            return
        self.en_cours = libelle
        if cle == 'poignet':
            command = 'wrist'
            parameters = {'target_angle_rad': math.pi / 2.0 if valeur else 0.0}
        else:
            command = 'suction'
            parameters = {
                'group': 'all' if cle == 'tous' else int(cle.removeprefix('etage_')),
                'enabled': bool(valeur),
            }
        decision_id, decision = self._record_choice(command, parameters)
        if decision is None:
            decision = self._operator_decision(command, parameters)
        self.operation_decision_id = decision_id
        self.demo_active_operation = {
            'id': decision_id or f'manual-{time.monotonic_ns()}',
            'type': decision['action'], 'status': 'running', 'cancelable': False,
            'failed': False, 'target_pose': None,
        }
        for b in self.boutons:
            b.state(['disabled'])
        self.ecrire(f'{libelle}…', 'info')
        self.pont.appeler(cle, valeur, libelle)

    # ------------------------------------------------------------------ carte de la base
    def echelle_carte(self):
        """Échelle (px/m) et décalage pour que l'arène tienne dans le canevas, proportions gardées."""
        w, h = self.carte.winfo_width(), self.carte.winfo_height()
        marge = 30
        lx = CARTE['x_max'] - CARTE['x_min']
        ly = CARTE['y_max'] - CARTE['y_min']
        k = max(min((w - 2 * marge) / lx, (h - 2 * marge) / ly), 1e-6)
        return k, (w - k * lx) / 2, (h - k * ly) / 2

    def vers_ecran(self, x, y):
        k, ox, oy = self.echelle_carte()
        return ox + (x - CARTE['x_min']) * k, oy + (CARTE['y_max'] - y) * k   # y vers le haut

    def vers_monde(self, u, v):
        k, ox, oy = self.echelle_carte()
        x = CARTE['x_min'] + (u - ox) / k
        y = CARTE['y_max'] - (v - oy) / k
        return (min(max(x, CARTE['x_min']), CARTE['x_max']),
                min(max(y, CARTE['y_min']), CARTE['y_max']))

    def dessiner_carte(self):
        c = self.carte
        c.delete('all')
        if c.winfo_width() < 50:
            return
        x0, y0 = self.vers_ecran(CARTE['x_min'], CARTE['y_max'])
        x1, y1 = self.vers_ecran(CARTE['x_max'], CARTE['y_min'])
        c.create_rectangle(x0, y0, x1, y1, fill=C['table'], outline='')

        pas = CARTE['pas']
        v = math.ceil(CARTE['x_min'] / pas) * pas
        while v <= CARTE['x_max'] + 1e-9:
            u, _ = self.vers_ecran(v, 0)
            c.create_line(u, y0, u, y1, fill=C['grille'])
            c.create_text(u, y1 + 4, text=f'{v:g}', anchor='n', fill=C['doux'], font=PETITE)
            v += pas
        v = math.ceil(CARTE['y_min'] / pas) * pas
        while v <= CARTE['y_max'] + 1e-9:
            _, w = self.vers_ecran(0, v)
            c.create_line(x0, w, x1, w, fill=C['grille'])
            c.create_text(x0 - 4, w, text=f'{v:g}', anchor='e', fill=C['doux'], font=PETITE)
            v += pas
        c.create_rectangle(x0, y0, x1, y1, outline=C['bordure'], width=2)
        safe_x0, safe_y1 = self.vers_ecran(
            CARTE['x_min'] + CARTE['marge_robot'], CARTE['y_max'] - CARTE['marge_robot']
        )
        safe_x1, safe_y0 = self.vers_ecran(
            CARTE['x_max'] - CARTE['marge_robot'], CARTE['y_min'] + CARTE['marge_robot']
        )
        c.create_rectangle(safe_x0, safe_y1, safe_x1, safe_y0,
                           outline=C['doux'], dash=(3, 3), width=1)

        if CARTE['x_min'] <= 0 <= CARTE['x_max'] and CARTE['y_min'] <= 0 <= CARTE['y_max']:
            u, w = self.vers_ecran(0, 0)
            c.create_line(u, w, u + 26, w, fill='#dc2626', width=2, arrow='last')
            c.create_line(u, w, u, w - 26, fill='#16a34a', width=2, arrow='last')

        if len(self.trace) >= 2:
            pts = [p for xy in self.trace for p in self.vers_ecran(*xy)]
            c.create_line(*pts, fill='#9aa4b2', width=2)
        if self.cible_pose:
            self.fleche(*self.cible_pose, plein=True)
        if self.pose_robot:
            self.dessiner_robot(*self.pose_robot)
        if self.saisie:
            self.fleche(*self.saisie, plein=False)

    def fleche(self, x, y, t, plein):
        u, v = self.vers_ecran(x, y)
        style = {} if plein else {'dash': (5, 3)}
        self.carte.create_line(u, v, u + 40 * math.cos(t), v - 40 * math.sin(t), fill=C['accent'],
                               width=3, arrow='last', arrowshape=(12, 14, 5), **style)
        self.carte.create_oval(u - 5, v - 5, u + 5, v + 5, outline=C['accent'], width=2,
                               fill=C['accent'] if plein else C['carte'])

    def dessiner_robot(self, x, y, t):
        c = self.carte
        s = CARTE['taille_robot'] / 2
        cos_t, sin_t = math.cos(t), math.sin(t)
        pts = []
        for dx, dy in ((s, s), (-s, s), (-s, -s), (s, -s)):
            pts += self.vers_ecran(x + dx * cos_t - dy * sin_t, y + dx * sin_t + dy * cos_t)
        c.create_polygon(*pts, fill='#dbe4f3', outline=C['texte'], width=2)
        u, v = self.vers_ecran(x, y)
        k = self.echelle_carte()[0]
        c.create_line(u, v, u + 1.3 * s * k * cos_t, v - 1.3 * s * k * sin_t,
                      fill=C['texte'], width=2, arrow='last')
        c.create_text(6, 4, anchor='nw', fill=C['doux'], font=PETITE,
                      text=f'robot   x {x:.2f}   y {y:.2f}   θ {math.degrees(t):.0f}°')

    def carte_survol(self, e):
        x, y = self.vers_monde(e.x, e.y)
        self.lbl_curseur.configure(text=f'x {x:.2f}   y {y:.2f} m')

    def carte_appui(self, e):
        x, y = self.vers_monde(e.x, e.y)
        marge = CARTE['marge_robot']
        x = min(max(x, CARTE['x_min'] + marge), CARTE['x_max'] - marge)
        y = min(max(y, CARTE['y_min'] + marge), CARTE['y_max'] - marge)
        # sans glisser, on garde l'orientation de la cible précédente (ou celle du robot)
        t = self.cible_pose[2] if self.cible_pose else (self.pose_robot[2] if self.pose_robot else 0.0)
        self.saisie = [x, y, t]
        self.appui_px = (e.x, e.y)
        self.dessiner_carte()

    def carte_glisse(self, e):
        if not self.saisie:
            return
        du, dv = e.x - self.appui_px[0], e.y - self.appui_px[1]
        if math.hypot(du, dv) > 6:
            self.saisie[2] = math.atan2(-dv, du)
            self.lbl_curseur.configure(text=f'θ {math.degrees(self.saisie[2]):.0f}°')
        self.dessiner_carte()

    def carte_relache(self, _e):
        if not self.saisie:
            return
        self.cible_pose = tuple(self.saisie)
        self.saisie = None
        self.envoyer_cible()

    def envoyer_cible(self):
        if not self.cible_pose:
            return
        x, y, t = self.cible_pose
        t_deg = math.degrees(t)
        self.lbl_cible.configure(text=f'Cible :  x {x:.2f} m    y {y:.2f} m    θ {t_deg:.0f}°')
        champs = {'x': x, 'y': y, 'theta': t}
        libelle = f'Aller à ({x:.2f} m, {y:.2f} m, {t_deg:.0f}°)'
        if self.action_en_cours == 'go_to_pose':
            # comme RViz : la nouvelle cible remplace l'ancienne
            self.annuler('go_to_pose')
            self.but_en_attente = (champs, libelle)
            self.ecrire('Nouvelle cible : elle part dès l\'arrêt du déplacement en cours', 'info')
        elif self.action_en_cours:
            self.ecrire('Centrage en cours : annule-le avant d\'envoyer une pose', 'erreur')
        else:
            self.lancer_action('go_to_pose', champs, libelle)
        self.dessiner_carte()

    def lancer_action(self, cle, champs, libelle):
        if self.action_en_cours or self.en_cours:
            self.ecrire('Attends la fin de l’opération en cours avant de déplacer le robot', 'erreur')
            return
        self.action_en_cours = cle
        if cle == 'go_to_pose':
            command = 'move'
            parameters = dict(champs)
        else:
            command = 'rock_centering'
            parameters = {}
        decision_id, decision = self._record_choice(command, parameters)
        if decision is None:
            decision = self._operator_decision(command, parameters)
        self.operation_decision_id = decision_id
        self.demo_active_operation = {
            'id': decision_id or f'manual-{time.monotonic_ns()}',
            'type': decision['action'], 'status': 'issuing',
            'cancelable': cle == 'go_to_pose', 'failed': False,
            'target_pose': decision.get('target_pose'),
        }
        self.btn_aller.state(['disabled'])
        self.btn_centrer.state(['disabled'])
        self.annulers[cle].state(['!disabled'])
        self.progres[cle].start(12)
        if cle == 'go_to_pose':
            self.lbl_nav_msg.configure(text='Envoi…')
            self.lbl_nav_t.configure(text='')
        else:
            self.lbl_marqueur.configure(text='●  Marqueur : —', foreground=C['doux'])
            for lbl in self.lbl_cen.values():
                lbl.configure(text='—')
            self.dessiner_cible()
        self.ecrire(f'{libelle}…', 'info')
        self.pont.envoyer_action(cle, champs, libelle)

    def annuler(self, cle):
        if cle == 'go_to_pose':
            self.but_en_attente = None
        if self.action_en_cours != cle:
            return
        operation_type = 'MOVE' if cle == 'go_to_pose' else 'ROCK_CENTERING'
        cancel_id, _ = self._record_choice('cancel', {'operation_type': operation_type})
        self.cancel_decision_id = cancel_id
        if self.demo_active_operation is not None:
            self.demo_active_operation['status'] = 'canceling'
            self.demo_active_operation['cancelable'] = False
        self.annulers[cle].state(['disabled'])
        self.ecrire('Annulation demandée…', 'info')
        self.pont.annuler(cle)

    def fin_action(self, cle, status, texte):
        action = (self.demo_active_operation or {}).get(
            'type', 'MOVE' if cle == 'go_to_pose' else 'UNSUPPORTED_ROCK_CENTERING'
        )
        self._record_result(self.operation_decision_id, action, status, texte)
        if self.cancel_decision_id is not None:
            cancel_status = 'succeeded' if status == 'canceled' else 'failed'
            self._record_result(self.cancel_decision_id, 'CANCEL', cancel_status, texte)
        self.operation_decision_id = None
        self.cancel_decision_id = None
        self.demo_active_operation = None
        ok = status == 'succeeded'
        self.ecrire(texte, 'ok' if ok else 'erreur')
        self.action_en_cours = None
        if self.cible_pose:
            self.btn_aller.state(['!disabled'])
        self.btn_centrer.state(['!disabled'])
        for b in self.annulers.values():
            b.state(['disabled'])
        for p in self.progres.values():
            p.stop()
        if cle == 'go_to_pose':
            self.lbl_nav_msg.configure(text=texte, foreground=C['ok'] if ok else C['erreur'])
            if self.but_en_attente:
                champs, libelle = self.but_en_attente
                self.but_en_attente = None
                self.lancer_action('go_to_pose', champs, libelle)

    def sur_feedback(self, cle, fb):
        if cle == 'go_to_pose':
            self.lbl_nav_msg.configure(text=fb.message or 'En route…', foreground=C['doux'])
            self.lbl_nav_t.configure(text=f'{fb.elapsed:5.1f} s')
            return
        vis = fb.marker_visible
        self.lbl_marqueur.configure(text='●  Marqueur visible' if vis else '●  Marqueur perdu',
                                    foreground=C['ok'] if vis else C['erreur'])
        self.lbl_cen['err_fwd'].configure(text=f'{fb.err_fwd * 1000:7.1f} mm')
        self.lbl_cen['err_lat'].configure(text=f'{fb.err_lat * 1000:7.1f} mm')
        self.lbl_cen['err_theta'].configure(text=f'{math.degrees(fb.err_theta):7.1f} °')
        self.lbl_cen['elapsed'].configure(text=f'{fb.elapsed:7.1f} s')
        self.dessiner_cible(fb.err_fwd, fb.err_lat, fb.err_theta, vis)

    # ------------------------------------------------------------------ boucle
    def ecrire(self, texte, tag):
        self.journal.configure(state='normal')
        self.journal.insert('end', time.strftime('%H:%M:%S ') + texte + '\n', tag)
        self.journal.see('end')
        self.journal.configure(state='disabled')

    def lire_file(self):
        try:
            while True:
                evt = self.file.get_nowait()
                if evt[0] == 'fin':
                    _, _, status, texte = evt
                    action = (self.demo_active_operation or {}).get('type', 'UNKNOWN')
                    self._record_result(self.operation_decision_id, action, status, texte)
                    self.operation_decision_id = None
                    self.demo_active_operation = None
                    ok = status == 'succeeded'
                    self.ecrire(texte, 'ok' if ok else 'erreur')
                    self.en_cours = None
                    for b in self.boutons:
                        b.state(['!disabled'])
                elif evt[0] == 'joints':
                    _, asc, poi = evt
                    if asc is not None:
                        self.lbl_asc.configure(text=f'{asc * 1000:6.1f} mm')
                    if poi is not None:
                        self.lbl_poi.configure(text=f'{math.degrees(poi):6.1f} °')
                elif evt[0] == 'fb':
                    self.sur_feedback(evt[1], evt[2])
                elif evt[0] == 'action_fin':
                    _, cle, status, texte = evt
                    self.fin_action(cle, status, texte)
                elif evt[0] == 'action_started':
                    if (self.action_en_cours == evt[1] and self.demo_active_operation is not None
                            and self.demo_active_operation.get('status') == 'issuing'):
                        self.demo_active_operation['status'] = 'running'
                elif evt[0] == 'info':
                    self.ecrire(evt[1], 'info')
                elif evt[0] == 'snapshot':
                    self._receive_observation(evt[1])
                elif evt[0] == 'observer_info':
                    self.ecrire(evt[1], 'info')
                elif evt[0] == 'observer_error':
                    self.ecrire(f'Observateur : {evt[1]}', 'erreur')
                elif evt[0] == 'observer_stopped':
                    if self.recording_starting:
                        self.recording_starting = False
                        self.btn_start_recording.state(['!disabled'])
                        self.lbl_recording.configure(text='Observateur arrêté', foreground=C['erreur'])
                    self.ecrire(f'Observateur terminé (code {evt[1]})', 'erreur')
                elif evt[0] == 'pose':
                    _, x, y, lacet = evt
                    self.pose_robot = (x, y, lacet)
                    if not self.trace or math.hypot(x - self.trace[-1][0], y - self.trace[-1][1]) > 0.01:
                        self.trace = (self.trace + [(x, y)])[-500:]
                    self.dessiner_carte()
        except queue.Empty:
            pass
        self.racine.after(50, self.lire_file)

    def verifier_services(self):
        dispo = self.pont.disponibilites()
        manquants = [nom for nom, ok in dispo.items() if not ok]
        if manquants:
            self.lbl_srv.configure(text='Absents : ' + ', '.join(manquants), foreground=C['erreur'])
        else:
            self.lbl_srv.configure(text='Tous les services et actions sont disponibles', foreground=C['ok'])
        self.racine.after(1000, self.verifier_services)

    def fermer(self):
        """Annule les actions en cours avant de fermer, pour ne pas laisser la base partir seule."""
        self.stop_recording(reason='operator console closed')
        for recorder in set(self.decision_recorders.values()):
            recorder.abort_pending(reason='operator console closed before a terminal result')
        self.decision_recorders.clear()
        if self.pont.annuler_tout():
            self.ecrire('Fermeture : annulation des actions en cours…', 'info')
            self.racine.after(400, self.racine.destroy)
        else:
            self.racine.destroy()


def main():
    rclpy.init()
    file_ros = queue.Queue()
    pont = Pont(file_ros)
    executeur = SingleThreadedExecutor()
    executeur.add_node(pont)
    fil = threading.Thread(target=executeur.spin, daemon=True)
    fil.start()

    racine = tk.Tk()
    Interface(racine, pont, file_ros)
    try:
        racine.mainloop()
    finally:
        executeur.shutdown()
        pont.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
