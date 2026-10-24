# ARCL Simulation — Eurobot 2027

Simulation Gazebo Harmonic / ROS 2 Jazzy de la table Eurobot 2027 et du robot holonome de l'ARCL.

## Lancer la simulation

La simulation nécessite aussi `robot_localization` :

```bash
sudo apt install ros-jazzy-robot-localization
```

```bash
./build.sh
source install/setup.bash
ros2 launch arcl_simulation sim.launch.py
```

Le launch démarre Gazebo, RViz, les nœuds du robot et le **pupitre** de commande.

## Pupitre

Interface graphique (Tkinter) pour piloter le robot sans taper de commande : [move.launch.py](src/arcl_simulation/launch/move.launch.py). Elle est lancée automatiquement par `sim.launch.py`, mais peut aussi être ouverte seule :

```bash
sudo apt install python3-tk   # une seule fois
python3 src/arcl_simulation/launch/move.launch.py
```

- **Ventouses** : les 3 étages ensemble ou un par un.
- **Poignet** : 0° ou 90° (cartons debout).
- **Aller à une pose** : clic sur la carte de la table pour la position, glisser pour l'orientation (envoie l'action `/go_to_pose`). Une nouvelle cible remplace la précédente.
- **Centrage sur une pierre** : lance `/rock_centering` et affiche les erreurs en direct.
- Affiche la position du robot (TF `map` → `base_link`), de l'ascenseur et du poignet (`/joint_states`).

Les actions en cours sont annulées à la fermeture de la fenêtre.

## Commander le robot

Le robot se commande uniquement par actions et services.

**Aller à une pose** (x, y en mètres, theta en radians) :

```bash
ros2 action send_goal /go_to_pose robot_msgs/action/GoToPose "{x: 1.0, y: 1.0, theta: 1.57}" --feedback
```

**Se centrer sur une pierre** (tag ArUco visible par la caméra) :

```bash
ros2 action send_goal /rock_centering robot_msgs/action/RockCentering "{}" --feedback
```

**Attraper / lâcher avec les ventouses** :

```bash
ros2 service call /robot_omni/ventouse/set std_srvs/srv/SetBool "{data: true}"   # attraper
ros2 service call /robot_omni/ventouse/set std_srvs/srv/SetBool "{data: false}"  # lâcher
```

Un seul étage : `/robot_omni/ventouse_<n>/set` (n = 1, 2, 3). Poignet : `/robot_omni/poignet/set` (`true` = 90°).

Détails techniques : [compilation](doc/compilation.md) et [ventouses](doc/ventouse.md).

## Stratégie

Le nœud `strategy` ([strategy.cpp](src/arcl_simulation/src/strategy.cpp)) est une machine à état qui joue une séquence de match en enchaînant les actions et services ci-dessus. Il n'est pas lancé par `sim.launch.py` :

```bash
ros2 run arcl_simulation strategy                                   # côté jaune
ros2 run arcl_simulation strategy --ros-args -p bleu:=true          # côté bleu (miroir x -> 3 - x)
ros2 run arcl_simulation strategy --ros-args -p duree_match:=100.0  # arrêt au bout de 100 s
```

La séquence est une liste d'étapes écrite en dur dans `actions` :

- `MOVE_TO` : `/go_to_pose` vers (x, y, theta en degrés).
- `CENTRAGE` : `/rock_centering` sur la pile en (x, y). S'il réussit, le `MOVE_TO` suivant est remplacé par la pose corrigée.
- `ROTATION_VENTOUSE` : poignet à 0° ou 90°.
- `VENTOUSE` : prendre / lâcher, les 3 étages ou un seul.

Chaque étape est rejouée en cas d'échec (2 essais), puis la séquence est abandonnée. `avance_apres_centrage` règle l'approche après le centrage (< 0 : calculée sur la carte, 0 : aucune, > 0 : avance fixe en m).

## Localisation

```
/joint_states ──> wheel_odometry ──> /wheel/odom ──┐
                                                  ├─> ekf_odom (robot_localization) ──> /odometry/filtered ──┐
pose vraie ─────> otos_simulator ──> /otos/odom ──┘                                                          ├─> beacon_localization ──> /pose_estimation
/scan (balises) ─────────────────────────────────────────────────────────────────────────────────────────────┘
```

- `wheel_odometry` : odométrie des 4 roues omni (vitesse + covariance, gonflée en cas de glissement détecté).
- `otos_simulator` : simule le SparkFun OTOS (erreur d'échelle, bruit, biais gyro, perte de suivi > 2,5 m/s). Paramètres réglables (`linear_scale_error`, `gyro_noise_density`, ...).
- `ekf_odom` : `ekf_node` de robot_localization, fusionne les vitesses roues + OTOS dans le repère `odom` ([config/ekf_odom.yaml](src/arcl_simulation/config/ekf_odom.yaml)).
- `beacon_localization` : EKF dans le repère `map`, prédiction avec `/odometry/filtered`, recalage sur les 3 balises vues au lidar.

## Calcul déporté (caméra fixe)

Une simulation de base du calcul déporté est disponible dans le paquet [deported_base_calculus](src/deported_base_calculus) (`camera_calculus`). Une caméra fixe sur un mât en bord de table (~1,8 m) publie son image sur `/deported_camera/image`, un `aruco_node` y détecte les tags et `cam_tracking` reçoit les marqueurs.

```
/deported_camera/image ──> aruco_node ──> /deported_aruco_markers ──> cam_tracking
```

```bash
ros2 launch deported_base_calculus cam.launch.py   # avec la simulation déjà lancée
```

Pour l'instant `cam_tracking` se contente d'afficher le nombre de marqueurs détectés : c'est la base sur laquelle construire le suivi des robots et des éléments de jeu. Paramètres ArUco : [config/aruco_parameters.yaml](src/deported_base_calculus/config/aruco_parameters.yaml).
