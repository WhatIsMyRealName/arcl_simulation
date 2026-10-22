// Ventouses génériques pour Gazebo Harmonic (gz-sim 8).
// Sur commande, colle au robot chaque objet (non statique) qui touche une des ventouses,
// puis lève l'ascenseur. Au lâcher : l'ascenseur redescend, puis les objets sont relâchés.
//
//   ros2 service call /robot_omni/ventouse/set std_srvs/srv/SetBool '{data: true}'
//   ros2 service call /robot_omni/ventouse/set std_srvs/srv/SetBool '{data: false}'
//
// Paramètres SDF (dans le <model> du robot) :
//   <link>      link qui porte les ventouses               (défaut : chassis)
//   <collision> collision d'une bouche de ventouse, répétable (au moins une)
//   <topic_commande> topic StringMsg des commandes identifiées
//   <topic_resultat> topic StringMsg des résultats finaux
//   <topic_ascenseur> topic gz.msgs.Double du JointPositionController de l'ascenseur (optionnel)
//   <hauteur>   hauteur de levée en m                      (défaut : 0.05)
//   <delai_lacher> temps de descente avant de relâcher, s  (défaut : 0.7)
//   <fenetre_prise> après le premier contact, temps pendant lequel on continue
//                  d'aspirer avant de lever, s (défaut : 0). Permet de coller
//                  plusieurs objets qui ne touchent pas tous au même pas.
//   <portee>    distance max, en m, entre le centre d'une bouche et la boîte de collision
//               d'un objet, côté bouche, pour le coller sans contact (défaut : 0 = contact
//               obligatoire). Le robot peut s'arrêter juste avant la pile sans la pousser.
//
// Une seule opération est acceptée à la fois, sans file d'attente.
// Commande : 1 <id> <PING|TAKE|RELEASE> <expiration_unix_ms>
// Résultat : 1 <id> <OK|ERROR> <code>
// L'échéance réelle est vérifiée même lorsque la simulation est en pause.
// État de diagnostic sur /model/<robot>/ventouse/etat (configurable).

#include <algorithm>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <sstream>
#include <string>
#include <vector>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>

#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Static.hh>
#include <sdf/Box.hh>
#include <sdf/Geometry.hh>

using namespace gz;
using namespace sim;

class Ventouse : public System, public ISystemConfigure, public ISystemPreUpdate
{
  public: ~Ventouse() override
  {
    if (!this->topicCommande.empty())
      this->node.Unsubscribe(this->topicCommande);
  }

  public: void Configure(const Entity &_entity,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         EntityComponentManager &_ecm, EventManager &) override
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->modele = _entity;
    Model model(_entity);
    const auto nomLink = _sdf->Get<std::string>("link", "chassis").first;
    const auto base = "/model/" + model.Name(_ecm) + "/ventouse";
    const auto topic = _sdf->Get<std::string>("topic_commande", base + "/commande").first;
    this->topicCommande = topic;
    this->pubResultat = this->node.Advertise<msgs::StringMsg>(
        _sdf->Get<std::string>("topic_resultat", base + "/resultat").first);
    this->pubEtat = this->node.Advertise<msgs::StringMsg>(
        _sdf->Get<std::string>("topic_etat", base + "/etat").first);
    // Même un plugin mal configuré doit pouvoir répondre NOT_READY.
    this->node.Subscribe(topic, &Ventouse::SurCommande, this);

    this->link = model.LinkByName(_ecm, nomLink);
    if (this->link == kNullEntity)
    {
      gzerr << "[Ventouse] link [" << nomLink << "] introuvable\n";
      return;
    }

    for (auto e = _sdf->FindElement("collision"); e; e = e->GetNextElement("collision"))
    {
      const auto nomCol = e->Get<std::string>();
      const Entity col = Link(this->link).CollisionByName(_ecm, nomCol);
      if (col == kNullEntity)
      {
        gzerr << "[Ventouse] collision [" << nomCol << "] introuvable\n";
        continue;
      }
      // Demande au moteur physique de remplir les contacts de cette collision.
      if (!_ecm.Component<components::ContactSensorData>(col))
        _ecm.CreateComponent(col, components::ContactSensorData());
      this->bouches.push_back(col);
    }
    if (this->bouches.empty())
    {
      gzerr << "[Ventouse] aucune <collision> valide\n";
      return;
    }

    this->hauteur = _sdf->Get<double>("hauteur", 0.05).first;
    this->delaiLacher = _sdf->Get<double>("delai_lacher", 0.7).first;
    this->fenetrePrise = std::max(0.0, _sdf->Get<double>("fenetre_prise", 0.0).first);
    this->portee = std::max(0.0, _sdf->Get<double>("portee", 0.0).first);
    if (_sdf->HasElement("topic_ascenseur"))
    {
      this->pubAscenseur = this->node.Advertise<msgs::Double>(
          _sdf->Get<std::string>("topic_ascenseur"));
      this->avecAscenseur = true;
    }

    gzmsg << "[Ventouse] " << this->bouches.size()
          << " ventouse(s) prête(s), commande sur " << topic << "\n";
    this->ok = true;
    this->PublierEtat();
  }

  private: enum class Etat { LIBRE, ASPIRATION, TENU, DESCENTE };

  private: struct Operation
  {
    std::string id;
    bool attraper;
    std::chrono::steady_clock::time_point echeance;
    bool commencee{false};
  };

  private: void PublierResultat(const std::string &_id, bool _success,
                               const std::string &_code)
  {
    msgs::StringMsg msg;
    msg.set_data("1 " + _id + (_success ? " OK " : " ERROR ") + _code);
    this->pubResultat.Publish(msg);
  }

  // Appelée uniquement avec le mutex détenu, après mise à jour de l'état physique.
  private: void TerminerOperation(bool _success, const std::string &_code)
  {
    const auto id = this->operation->id;
    this->operation.reset();
    this->PublierResultat(id, _success, _code);
  }

  private: void SurCommande(const msgs::StringMsg &_msg)
  {
    std::lock_guard<std::mutex> l(this->mutex);
    std::istringstream input(_msg.data());
    std::string version, id, commande, extra;
    int64_t expiration;
    if (!(input >> version >> id))
    {
      gzerr << "[Ventouse] commande sans identifiant\n";
      return;
    }
    if (version != "1" || !(input >> commande >> expiration) || (input >> extra) ||
        (commande != "PING" && commande != "TAKE" && commande != "RELEASE"))
    {
      this->PublierResultat(id, false, "INVALID_COMMAND");
      return;
    }
    if (!this->ok)
    {
      this->PublierResultat(id, false, "NOT_READY");
      return;
    }
    if (commande == "PING")
    {
      this->PublierResultat(id, true, "READY");
      return;
    }
    if (this->operation)
    {
      this->PublierResultat(id, false, "BUSY");
      return;
    }
    const auto maintenant = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (expiration <= maintenant)
    {
      this->PublierResultat(id, false, "TIMEOUT");
      return;
    }
    // Borne le protocole et évite tout débordement lors de la conversion d'horloge.
    if (expiration > maintenant + 3600000)
    {
      this->PublierResultat(id, false, "INVALID_COMMAND");
      return;
    }
    this->operation = Operation{id, commande == "TAKE",
        std::chrono::steady_clock::now() + std::chrono::milliseconds(expiration - maintenant)};
  }

  public: void PreUpdate(const UpdateInfo &_info, EntityComponentManager &_ecm) override
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    if (!this->ok)
      return;
    const auto now = std::chrono::steady_clock::now();
    if (this->dernierHeartbeat == std::chrono::steady_clock::time_point{} ||
        now - this->dernierHeartbeat >= std::chrono::seconds(1))
      this->PublierEtat();
    if (!this->operation)
      return;
    if (std::chrono::steady_clock::now() >= this->operation->echeance)
    {
      if (this->etat == Etat::ASPIRATION)
      {
        this->Detacher(_ecm);   // prise partielle pendant la fenêtre : on annule tout
        this->ChangerEtat(Etat::LIBRE);
      }
      else if (this->etat == Etat::DESCENTE)
      {
        this->Ascenseur(this->hauteur);
        this->ChangerEtat(Etat::TENU);  // liaisons conservées
      }
      this->TerminerOperation(false, "TIMEOUT");
      return;
    }
    // Les commandes idempotentes ne nécessitent aucun pas de physique.
    if (!this->operation->commencee)
    {
      if (this->operation->attraper && this->etat == Etat::TENU)
      {
        this->TerminerOperation(true, "ALREADY_HELD");
        return;
      }
      if (!this->operation->attraper && this->etat == Etat::LIBRE)
      {
        this->TerminerOperation(true, "ALREADY_FREE");
        return;
      }
    }
    if (_info.paused)
      return;
    const double t = std::chrono::duration<double>(_info.simTime).count();

    if (!this->operation->commencee)
    {
      this->operation->commencee = true;
      if (this->operation->attraper)
      {
        this->ChangerEtat(Etat::ASPIRATION);
        gzmsg << "[Ventouse] aspiration, contacts : " << this->DecrireContacts(_ecm) << "\n";
      }
      else
      {
        this->Ascenseur(0.0);
        this->lacherA = this->avecAscenseur ? t + this->delaiLacher : t;
        this->ChangerEtat(Etat::DESCENTE);
      }
    }

    if (this->etat == Etat::DESCENTE)
    {
      if (t < this->lacherA)
        return;
      this->Detacher(_ecm);
      gzmsg << "[Ventouse] lâché\n";
      this->ChangerEtat(Etat::LIBRE);
      this->TerminerOperation(true, "RELEASED");
      return;                 // contacts à jour au pas suivant
    }

    // Aspiration active : on colle tout ce qui touche, à chaque pas, pendant la fenêtre
    // qui suit le premier contact, puis on lève.
    if (this->etat == Etat::ASPIRATION && this->Attraper(_ecm))
    {
      if (this->premierContact < 0)
        this->premierContact = t;
      if (t >= this->premierContact + this->fenetrePrise)
      {
        gzmsg << "[Ventouse] " << this->attaches.size() << " objet(s) tenu(s)\n";
        this->premierContact = -1;
        this->Ascenseur(this->hauteur);
        this->ChangerEtat(Etat::TENU);
        this->TerminerOperation(true, "TAKEN");
      }
    }
  }

  // Diagnostic : liste de ce que touchent les ventouses.
  private: std::string DecrireContacts(EntityComponentManager &_ecm)
  {
    std::set<std::string> noms;
    for (const Entity bouche : this->bouches)
    {
      auto contacts = _ecm.Component<components::ContactSensorData>(bouche);
      if (!contacts)
        continue;
      for (const auto &c : contacts->Data().contact())
      {
        const Entity autre = c.collision1().id() == bouche ?
            c.collision2().id() : c.collision1().id();
        auto nom = _ecm.Component<components::Name>(topLevelModel(autre, _ecm));
        noms.insert(nom ? nom->Data() : "?");
      }
    }
    if (noms.empty())
      return "aucun";
    std::string r;
    for (const auto &n : noms)
      r += (r.empty() ? "" : ", ") + n;
    return r;
  }

  private: std::string EtatTexte() const
  {
    return this->etat == Etat::LIBRE ? "libre" :
           this->etat == Etat::ASPIRATION ? "aspiration" :
           this->etat == Etat::TENU ? "tenu" : "descente";
  }

  private: void PublierEtat()
  {
    msgs::StringMsg m;
    m.set_data(this->EtatTexte());
    this->pubEtat.Publish(m);
    this->dernierHeartbeat = std::chrono::steady_clock::now();
  }

  private: void ChangerEtat(Etat _e)
  {
    this->etat = _e;
    this->PublierEtat();
  }

  private: void Ascenseur(double _z)
  {
    if (!this->avecAscenseur)
      return;
    msgs::Double m;
    m.set_data(_z);
    this->pubAscenseur.Publish(m);
  }

  // Retire toutes les liaisons créées et oublie les objets tenus.
  private: void Detacher(EntityComponentManager &_ecm)
  {
    for (const Entity j : this->liaisons)
      _ecm.RequestRemoveEntity(j);
    this->liaisons.clear();
    this->attaches.clear();
    this->premierContact = -1;
  }

  // Objets à portée : boîte de collision à moins de <portee> du centre d'une bouche,
  // devant la bouche. Aucun contact n'est nécessaire, donc la pile n'est pas poussée.
  private: void APortee(EntityComponentManager &_ecm, std::set<Entity> &_touches)
  {
    if (this->portee <= 0.0)
      return;
    std::vector<std::pair<math::Vector3d, math::Vector3d>> centres;  // centre, axe de la bouche
    for (const Entity b : this->bouches)
    {
      const math::Pose3d p = worldPose(b, _ecm);
      centres.emplace_back(p.Pos(), p.Rot().RotateVector(math::Vector3d::UnitZ));
    }
    _ecm.Each<components::Collision, components::Geometry, components::ParentEntity>(
      [&](const Entity &_col, const components::Collision *,
          const components::Geometry *_geom, const components::ParentEntity *_parent) -> bool
      {
        const sdf::Box *boite = _geom->Data().BoxShape();
        if (_geom->Data().Type() != sdf::GeometryType::BOX || !boite)
          return true;
        const Entity lien = _parent->Data();
        const Entity modeleAutre = topLevelModel(lien, _ecm);
        if (modeleAutre == this->modele)
          return true;                      // soi-même
        auto stat = _ecm.Component<components::Static>(modeleAutre);
        if (stat && stat->Data())
          return true;                      // mur, balise...
        const math::Pose3d pc = worldPose(_col, _ecm);
        const math::Vector3d demi = boite->Size() / 2.0;
        for (const auto &[centre, axe] : centres)
        {
          // point de la boîte le plus proche du centre de la bouche
          const math::Vector3d l = pc.Rot().RotateVectorReverse(centre - pc.Pos());
          const math::Vector3d proche(std::clamp(l.X(), -demi.X(), demi.X()),
                                      std::clamp(l.Y(), -demi.Y(), demi.Y()),
                                      std::clamp(l.Z(), -demi.Z(), demi.Z()));
          const math::Vector3d d = pc.Pos() + pc.Rot().RotateVector(proche) - centre;
          if (d.Length() <= this->portee && d.Dot(axe) >= 0.0)
          {
            _touches.insert(lien);
            break;
          }
        }
        return true;
      });
  }

  // Colle les nouveaux objets touchés ; vrai si au moins un objet est tenu.
  private: bool Attraper(EntityComponentManager &_ecm)
  {
    std::set<Entity> touches;   // links touchés, chacun attaché une seule fois
    for (const Entity bouche : this->bouches)
    {
      auto contacts = _ecm.Component<components::ContactSensorData>(bouche);
      if (!contacts)
        continue;
      for (const auto &c : contacts->Data().contact())
      {
        const Entity autre = c.collision1().id() == bouche ?
            c.collision2().id() : c.collision1().id();
        auto parent = _ecm.Component<components::ParentEntity>(autre);
        if (!parent)
          continue;
        const Entity linkAutre = parent->Data();
        const Entity modeleAutre = topLevelModel(linkAutre, _ecm);
        if (modeleAutre == this->modele)
          continue;                       // soi-même
        auto stat = _ecm.Component<components::Static>(modeleAutre);
        if (stat && stat->Data())
          continue;                       // mur, balise...
        touches.insert(linkAutre);
      }
    }
    this->APortee(_ecm, touches);

    for (const Entity l : touches)
    {
      if (!this->attaches.insert(l).second)
        continue;                         // déjà collé à un pas précédent
      const Entity j = _ecm.CreateEntity();
      _ecm.CreateComponent(j, components::DetachableJoint({this->link, l, "fixed"}));
      this->liaisons.push_back(j);
      auto nom = _ecm.Component<components::Name>(topLevelModel(l, _ecm));
      gzmsg << "[Ventouse] attrapé " << (nom ? nom->Data() : "?") << "\n";
    }
    return !this->attaches.empty();
  }

  private: transport::Node node;
  private: std::string topicCommande;
  private: transport::Node::Publisher pubAscenseur, pubEtat, pubResultat;
  private: bool avecAscenseur{false};
  private: double hauteur{0.05}, delaiLacher{0.7}, lacherA{-1};
  private: double fenetrePrise{0.0}, premierContact{-1}, portee{0.0};
  private: std::set<Entity> attaches;
  private: Entity modele{kNullEntity}, link{kNullEntity};
  private: std::vector<Entity> bouches, liaisons;
  private: std::mutex mutex;
  private: std::optional<Operation> operation;
  private: Etat etat{Etat::LIBRE};
  private: std::chrono::steady_clock::time_point dernierHeartbeat{};
  private: bool ok{false};
};

GZ_ADD_PLUGIN(Ventouse, System, ISystemConfigure, ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(Ventouse, "ventouse::Ventouse")
