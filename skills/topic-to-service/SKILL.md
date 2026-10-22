---
name: topic-to-service
description: Remplace un topic ROS 2 de commande par un service et met à jour les nœuds, dépendances, bridges Gazebo, fichiers launch, tests et documentation associés. À utiliser pour une commande ponctuelle avec une réponse finale ; garder un topic pour les flux continus et envisager une action ROS pour les opérations longues avec feedback.
---

# Convertir un topic de commande en service ROS 2

Utilise ce guide quand une commande aujourd’hui publiée sur un topic doit devenir une requête avec résultat. Suis la commande de bout en bout avant de modifier l’interface : nœud ROS, éventuel plugin ou backend, configuration du bridge, lancement, dépendances, tests et exemples d’utilisation.

## 1. Cartographier le topic existant

- Recherche le nom du topic, son type et les variantes du nom dans tout le dépôt. Repère chaque publisher, subscriber, paramètre SDF, entrée de bridge, déclaration launch, appelant, test et exemple documentaire.
- Note le comportement actuel : valeurs acceptées, effets physiques, ordre des commandes, traitement des demandes simultanées, état publié et gestion des erreurs.
- Distingue la commande du flux d’état. Un topic de télémétrie ou de feedback continu peut rester un topic même si la commande devient un service.

## 2. Définir le contrat du service

- Choisis un service existant si ses champs représentent exactement la requête et la réponse attendues. Par exemple, `std_srvs/srv/SetBool` convient à une commande booléenne qui renvoie `success` et `message`.
- Crée ou adapte une interface de service dédiée si la requête ou le résultat nécessite des champs typés supplémentaires. Mets alors à jour le paquet d’interfaces, sa génération ROS et les dépendances des paquets clients et serveurs.
- Définis les cas de réussite, les erreurs, les demandes idempotentes, les conflits de concurrence et les délais avant de coder. La réponse doit exprimer un résultat terminal compréhensible par l’appelant.
- Un service n’envoie pas de feedback intermédiaire. Pour une opération longue dont l’appelant doit suivre la progression ou demander l’annulation, évalue une action ROS plutôt qu’un service.
- Décide si le nom du topic public est supprimé ou s’il faut une période de compatibilité. Si l’API change, migre tous les appelants et retire l’ancien chemin seulement selon la compatibilité attendue du projet.

## 3. Choisir où se trouve l’adaptateur

Pour un serveur ROS 2 direct, crée le service dans le nœud qui possède le comportement et élimine le publisher de commande devenu inutile.

Pour un backend Gazebo qui reçoit déjà des commandes via `ros_gz_bridge`, garde le service comme API ROS publique dans un nœud adaptateur. Le nœud traduit la requête en commande backend puis associe la réponse backend à la requête ROS en attente. Ne suppose pas que le bridge topic de Gazebo transporte directement un service ROS.

Ne considère pas la disponibilité du service ROS comme une preuve que le bridge et le backend sont prêts : les actions de launch peuvent démarrer sans attendre leur découverte. Avant d’envoyer une commande, vérifie la disponibilité du backend avec un signal explicite ou réponds « indisponible ». Si une commande peut être publiée avant que son abonné soit découvert, prévois un accusé de réception et une reprise sûre, ou refuse-la ; une profondeur de file ou une QoS fiable ne conserve pas une publication volatile pour les abonnés qui se connectent plus tard. Vérifie aussi qu’une requête perdue ne laisse pas l’adaptateur occupé indéfiniment.

Dans les deux cas, vérifie qu’une requête n’est déclarée réussie qu’après le résultat défini par le contrat. Si le backend répond de façon asynchrone, corrèle chaque réponse à une requête, ignore les réponses périmées ou mal formées et décide explicitement du comportement au timeout et à la concurrence. Ne bloque pas le callback du service en attendant le backend : laisse l’exécuteur traiter les réponses et les délais, et protège l’état partagé selon le modèle d’exécution utilisé.

## 4. Mettre à jour le nœud et les imports

- En C++, ajoute l’include du service, par exemple `#include <std_srvs/srv/set_bool.hpp>`, puis le type `rclcpp::Service<...>` et le callback `create_service`. Garde les includes nécessaires aux topics internes si l’adaptateur publie une commande et écoute un résultat.
- En Python, importe le type avec `from std_srvs.srv import SetBool`, crée le serveur avec `create_service`, et mets à jour les clients avec `create_client` et leurs requêtes.
- Nomme et valide les champs de requête explicitement. Renseigne toujours tous les champs de réponse et retourne des erreurs métier sous une forme stable.
- Conserve les subscribers ou publishers de diagnostic utiles. Retire les membres, callbacks, types de message et includes de l’ancien topic de commande qui ne servent plus.

## 5. Mettre à jour les dépendances et l’intégration

- Dans `package.xml`, déclare le paquet de service en dépendance de build et d’exécution (par exemple `<depend>std_srvs</depend>`). Ajoute aussi les dépendances de test réellement utilisées.
- Dans le `CMakeLists.txt` concerné, ajoute `find_package(... REQUIRED)`, puis le paquet dans `ament_target_dependencies` de la cible qui inclut le service. Déclare et installe toute nouvelle cible exécutable ; garde le standard C++ requis par le code.
- Dans le launch, démarre le nœud serveur ou adaptateur avec le bon nom d’exécutable. Passe les paramètres requis et vérifie qu’il est lancé avec le backend dont il dépend.
- Mets à jour la configuration `ros_gz_bridge` en fonction des topics internes encore nécessaires : nom ROS, nom Gazebo, types ROS/Gazebo et direction. Retire l’ancien bridge de commande publique. Un service ROS fourni par le nœud adaptateur n’est pas une entrée YAML de bridge topic.
- Mets à jour le SDF/plugin pour ses topics internes de commande et de résultat, si le backend en utilise. Garde les topics d’état nécessaires au diagnostic et vérifie leurs noms de paramètres côté plugin et côté configuration.

## 6. Adapter les tests et les exemples

- Remplace les tests de l’API publique qui publiaient l’ancien topic par des appels au service. Vérifie le contrat observable : requête traduite, réponse correcte, cas d’erreur, concurrence, démarrage avec backend retardé et timeout si ces cas sont prévus. Garde les tests directs du plugin sur son interface interne quand ils vérifient le comportement Gazebo indépendamment du service.
- Pour un adaptateur asynchrone, utilise un backend factice afin de vérifier la corrélation des réponses, les réponses tardives et la réservation d’une requête en attente.
- Pour un plugin ou un backend physique simulé, garde des tests d’intégration du comportement réel : effets, ordre des opérations, état final et comportement en pause si la simulation peut être pausée.
- Mets à jour les dépendances de test dans `package.xml` et leur déclaration CMake/ament. Ne confonds pas le test du serveur ROS avec le test du backend.
- Remplace les commandes `ros2 topic pub` de commande par `ros2 service call`. Documente le nom complet, le type, la structure de requête, les réponses et les limites importantes. Enlève les instructions anciennes uniquement si l’ancien endpoint est effectivement retiré.

## Exemple du dépôt

La ventouse de `arcl_simulation` illustre un service public au-dessus d’un backend Gazebo : `/robot_omni/ventouse/set` utilise `std_srvs/srv/SetBool`, tandis que le nœud `ventouse_service` échange avec le plugin via les topics internes `/robot_omni/ventouse/commande` et `/robot_omni/ventouse/resultat`, configurés dans `src/arcl_simulation/config/ros_bridge.yaml`. Le launch démarre l’adaptateur, et le SDF déclare ses topics au plugin. Utilise ces fichiers comme exemples de structure, mais vérifie toujours les noms et le comportement actuels avant de les reprendre :

- `src/arcl_simulation/src/ventouse_service.cpp`
- `src/arcl_simulation/CMakeLists.txt` et `src/arcl_simulation/package.xml`
- `src/arcl_simulation/launch/sim.launch.py`
- `src/arcl_simulation/config/ros_bridge.yaml`
- `src/arcl_simulation/worlds/arene_omni.sdf` et `src/ventouse_plugin/Ventouse.cc`
- `src/ventouse_plugin/test/ventouse_integration.cc`

## Revue finale

Avant de conclure, recherche encore l’ancien topic pour trouver les usages résiduels. Vérifie que le service est construit, installé et lancé ; que tous ses types sont déclarés dans les imports et dépendances ; que le bridge ne conserve que les topics internes nécessaires ; et que le README décrit l’API réellement exposée. Signale les changements d’API incompatibles et les limites de validation constatées.
