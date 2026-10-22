# Service des ventouses

Le service renvoie uniquement le résultat final (`success`, `message`), sans feedback.
Une prise réussit dès qu'un objet est attaché et que la montée est commandée ; la
position de l'ascenseur n'est pas vérifiée. Un lâcher attend 0,7 seconde de simulation
avant de demander le détachement. Demander un état déjà atteint réussit sans mouvement.
Une demande concurrente est refusée avec « Ventouse occupée », sans mise en file.

## Disponibilité du backend

Avant chaque opération, le service vérifie le trajet aller-retour jusqu'au plugin
avec `PING` / `READY`, sans effet physique. Sans réponse pendant le délai du
watchdog, il renvoie « Backend de ventouse indisponible » et permet un nouvel
appel. Le délai de l'opération commence après cette confirmation.

## Délais et erreurs de transport

Le paramètre `operation_timeout_sec` du nœud `ventouse_service` vaut 2 secondes
réelles par défaut (valeurs autorisées : ]0, 3600]). La pause ne suspend pas ce délai.
À expiration, une prise arrête l'aspiration ; un lâcher conserve l'objet et commande
la remontée, qui reprendra avec la simulation. Le watchdog de transport vaut
`max(3, operation_timeout_sec + 1)` secondes réelles. Si aucun résultat n'arrive,
le service renvoie un échec mais reste réservé jusqu'au résultat terminal correspondant.
Si le backend disparaît définitivement, redémarrer l'ensemble backend/service pour
rétablir un état connu avant de reprendre les commandes.

## Protocole interne ROS / Gazebo

Le nœud ROS 2 échange avec le plugin Gazebo par deux topics internes, bridgés en
`std_msgs/msg/String` ↔ `gz.msgs.StringMsg` :

- `/robot_omni/ventouse/commande` : `1 <id> <PING|TAKE|RELEASE> <expiration_unix_ms>`.
- `/robot_omni/ventouse/resultat` : `1 <id> <OK|ERROR> <code>`.

Les codes sont `READY`, `TAKEN`, `RELEASED`, `ALREADY_HELD`, `ALREADY_FREE`, `BUSY`,
`TIMEOUT`, `INVALID_COMMAND` et `NOT_READY`. Les résultats anciens ou mal formés
sont ignorés par le service. Les deux processus doivent partager l'horloge système.
L'ancien topic booléen `/robot_omni/ventouse` est retiré. L'état Gazebo
`/robot_omni/ventouse/etat` reste un diagnostic interne. Sur le robot réel, un pilote
matériel peut remplacer le backend Gazebo en conservant le service public.

## Test d’intégration du plugin

Après compilation et chargement de `install/setup.bash` :

```bash
colcon test --merge-install --packages-select ventouse_plugin
```

Ce test C++ lance Gazebo sans interface graphique et vérifie le comportement du
plugin.
