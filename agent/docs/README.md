# Agent visuel : collecte, préparation et entraînement

Depuis la racine du dépôt, installer `agent/requirements.txt`, compiler avec
`./build.sh`, puis sourcer ROS et le workspace dans chaque terminal :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## 1. Simulation et collecte commune

```sh
ros2 launch arcl_simulation sim.launch.py rviz:=false command_events:=true
```

Dans un autre terminal :

```sh
python3 -m agent.architecture.collect_ros --output runs/sessions/demo-01 --carton-source gazebo_sim
```

Piloter ensuite avec le pupitre, la stratégie C++ ou un autre client ROS. Arrêter
la collecte par Ctrl+C. Le pupitre et la stratégie fonctionnent indépendamment
de l’agent ; la collecte se lance dans un terminal séparé. `record_strategy` est
un alias du collecteur commun.

La télémétrie est désactivée par défaut. `command_events:=true` configure le
paramètre de démarrage `publish_command_events` sur les deux serveurs. Pour les
exécutables lancés directement, ajouter `--ros-args -p publish_command_events:=true`.
Ce paramètre est en lecture seule ; il faut redémarrer les serveurs pour changer
ce choix. Sans activation, aucun publisher d’événements ni suivi de requêtes
n’est créé. L’observateur refuse de produire des observations si les serveurs
n’ont pas activé la télémétrie. `--command-event-nodes` permet de préciser leurs
noms ROS lorsque l’on utilise un namespace ou des noms personnalisés.

Le collecteur observe les événements publiés par les serveurs et les messages
`/speed_command`. Un centrage produit donc des exemples `SPEED`, sans action
`CENTRAGE` apprise. Les commandes refusées sont conservées dans les événements
bruts et exclues des cibles.

Chaque session contient `manifest.json`, `events.jsonl`, `observations.jsonl` et
`images/*.jpg`. Les écritures sont progressives. Une session interrompue reste
exploitable, avec `complete: false`. La préparation ignore uniquement une dernière
ligne inachevée et signale une corruption intermédiaire.

Les rôles ROS autorisés sont configurables avec `--topics-config topics.json`,
par exemple `{"image":"/camera/image","pose":"/pose_estimation"}`. La caméra
déportée, `/rock_centering` et la pose vraie du robot sont exclus de ces sources.
Le fournisseur `gazebo_sim` conserve explicitement les positions simulées des
cartons pour le développement ; aucun fournisseur de caméra déportée réelle
n’est encore branché.

## 2. Préparer les cibles

```sh
python3 -m agent.architecture.prepare_ros runs/sessions/demo-01 --output runs/targets/demo-01.jsonl
```

Le fichier `.preparation.json` contient les effectifs retenus/exclus. Les images
sont référencées depuis le fichier de cibles ; conserver les sessions sources.
Collecter plusieurs sessions indépendantes pour la séparation train/validation.

## 3. Entraînement

```sh
python3 -m agent.architecture.train_targets runs/targets --output checkpoints/policy.pt --epochs 20
```

Ajouter `--model checkpoints/policy.pt` pour reprendre le modèle courant. Chaque
entraînement crée un dossier sous `runs/training/` : configuration, empreintes des
jeux de données, partitions par session, journal texte, métriques CSV/JSON/JSONL et
`best.pt`. `--run-directory` permet de choisir un nouveau dossier explicitement.
Le terminal affiche une ligne lisible par époque. Les images sont chargées à la
demande et encodées par lots. Les exemples visuels ne sont pas réfléchis par
l’augmentation de symétrie ; les exemples sans image le restent.

Les paramètres `--speed-linear-limit` (défaut 0,25 m/s) et
`--speed-angular-limit` (défaut 1 rad/s) sont sauvegardés avec le modèle. Configurer
les mêmes limites sur `robot_control` via ses paramètres ROS du même nom.

## 4. Exécution

```sh
python3 -m agent.architecture.world_observer --carton-source gazebo_sim \
  | python3 -m agent.architecture.run_agent --model checkpoints/policy.pt --records runs/live
```

L’observateur écrit les JPEG référencés sous `runs/observations/`. On peut choisir
`--image-directory`. Conserver ce dossier avec les journaux d’exécution.
Les flux JSON entre processus constituent le protocole de transport ; le journal
d’entraînement destiné au terminal reste lisible.

`MOVE` envoie une pose absolue avec orientation sin/cos à `/go_to_pose`. `SPEED`
prédit `(v, sin(direction), cos(direction), omega)` et publie les vitesses robot
`vx=v*cos(direction)`, `vy=v*sin(direction)`, `omega`. La direction de translation
et la rotation sont indépendantes. Chaque commande dure 100 ms ; les vitesses
expirent côté contrôleur après 200 ms sans renouvellement. Une image absente ou
vieille de plus de 200 ms masque `SPEED`.

Le contrôleur refuse les vitesses pendant un `go_to_pose` actif. Annuler la
navigation et attendre son résultat terminal avant de passer au mode relatif.
La fermeture, les erreurs d’observation et les pertes de flux arrêtent les
vitesses ; aucune action `/rock_centering` n’est exposée à l’agent.

## Self-play et vérifications

Le backend self-play reste fourni par l’utilisateur. Il doit fournir des
observations, des références d’images accessibles et exécuter `SPEED` sur
100 ms. La préparation conserve les cibles catégorielles issues des récompenses,
sans fabriquer de cibles supervisées de vitesse. Les règles de récompense et le
backend Gazebo RL ne sont pas implémentés par cette évolution.

```sh
python3 -m unittest discover -s agent/tests -v
```

Pour une simulation sans interfaces graphiques :

```sh
ros2 launch arcl_simulation sim.launch.py rviz:=false command_events:=true pupitre:=false \
  "gz_flags:=-s -r --headless-rendering"
```

Le lancement complet nécessite `robot_localization` et un moteur de rendu
compatible avec Gazebo pour la caméra. Le test ROS sans Gazebo peut être lancé
explicitement après build et chargement de l’environnement :

```sh
python3 -m agent.tests.ros_optional_validation
python3 -m agent.tests.ros_validation
```
