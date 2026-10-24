#include "command_events.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>

// Pilote les 3 étages de ventouses, l'ascenseur et le poignet.
//
//   /robot_omni/ventouse/set     (SetBool) : les 3 étages ensemble, true = prendre, false = lâcher
//   /robot_omni/ventouse_<n>/set (SetBool) : un seul étage (1 = carton du bas, poignet à 0°)
//   /robot_omni/poignet/set      (SetBool) : true = tourner à 90° (pierres debout), false = revenir à 0°
//
// Avant de tourner avec des pierres tenues, l'ascenseur monte au maximum (hauteur_rotation)
// pour que la pile ne touche pas le sol, puis il revient à la hauteur de transport.
//
// Prendre : les étages demandés collent ce qui est à portée, puis l'ascenseur monte.
// Lâcher  : l'ascenseur descend, on attend, les étages demandés lâchent, puis l'ascenseur
//           remonte si d'autres étages tiennent encore une pierre.
// Une seule demande de ventouses à la fois. Les callbacks sont sérialisés par le
// SingleThreadedExecutor et ne bloquent jamais.
class VentouseService : public rclcpp::Node
{
public:
  using SetBool = std_srvs::srv::SetBool;
  using Clock = std::chrono::steady_clock;
  static constexpr int N = 3;

  VentouseService() : Node("ventouse_service")
  {
    command_events_ = CommandEvents::create_if_enabled(this);
    timeout_ = duree(declare_parameter("operation_timeout_sec", 2.0), "operation_timeout_sec");
    delai_lacher_ = duree(declare_parameter("delai_lacher", 1.2), "delai_lacher");
    duree_poignet_ = duree(declare_parameter("duree_poignet", 1.5), "duree_poignet");
    hauteur_ = declare_parameter("hauteur", 0.05);
    hauteur_rotation_ = declare_parameter("hauteur_rotation", 0.12);
    duree_montee_ = duree(declare_parameter("duree_montee", 0.4), "duree_montee");
    angle_poignet_ = declare_parameter("angle_poignet", 1.5708);

    std::random_device random;
    std::ostringstream prefix;
    prefix << std::hex << random() << '-' << random() << '-'
           << Clock::now().time_since_epoch().count();
    prefix_ = prefix.str();

    ascenseur_pub_ = create_publisher<std_msgs::msg::Float64>("/robot_omni/ascenseur/cmd_pos", 10);
    poignet_pub_ = create_publisher<std_msgs::msg::Float64>("/robot_omni/poignet/cmd_pos", 10);

    for (int i = 0; i < N; ++i) {
      const std::string base = "/robot_omni/ventouse_" + std::to_string(i + 1);

      etages_[i].commande = create_publisher<std_msgs::msg::String>(base + "/commande", 10);
      
      etages_[i].resultat = create_subscription<std_msgs::msg::String>(
        base + "/resultat", 10,
        [this, i](const std_msgs::msg::String & msg) {on_result(i, msg);}
      );

      etages_[i].service = create_service<SetBool>(
        base + "/set",
        [this, i](std::shared_ptr<rmw_request_id_t> header, SetBool::Request::SharedPtr req) {
          on_request(etages_[i].service, std::move(header), {i}, req->data);
        }
      );
    }

    tous_ = create_service<SetBool>(
      "/robot_omni/ventouse/set",
      [this](std::shared_ptr<rmw_request_id_t> header, SetBool::Request::SharedPtr req) {
        on_request(tous_, std::move(header), {0, 1, 2}, req->data);
      }
    );

    service_poignet_ = create_service<SetBool>(
      "/robot_omni/poignet/set",
      [this](std::shared_ptr<rmw_request_id_t> header, SetBool::Request::SharedPtr req) {
        on_poignet(std::move(header), req->data);
      }
    );

    timer_ = create_wall_timer(std::chrono::milliseconds(20), [this]() {tick();});
  }

private:
  enum class Phase { PING, DESCENTE, COMMANDE };

  struct Etage{
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr commande;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr resultat;
    rclcpp::Service<SetBool>::SharedPtr service;
    bool tenu{false};
  };

  struct Demande{
    rclcpp::Service<SetBool>::SharedPtr service;
    std::shared_ptr<rmw_request_id_t> header;
    std::vector<int> etages;
    bool prendre;
    Phase phase{Phase::PING};
    Clock::time_point echeance, prochain_ping, fin_descente;
    std::array<std::string, N> ids;   // identifiant de la dernière commande envoyée par étage
    std::set<int> prets, finis;
    std::string erreur;               // vide si tout va bien
  };

  struct Rotation{
    std::shared_ptr<rmw_request_id_t> header;
    Clock::time_point debut, fin;   // debut : fin de la montée de l'ascenseur
    bool debout;
    bool lancee{false};
    bool charge{false};             // des pierres sont tenues
  };

  std::chrono::milliseconds duree(double s, const char * nom){
    if (!std::isfinite(s) || s <= 0.0 || s > 3600.0) {
      throw std::invalid_argument(std::string(nom) + " doit être dans ]0, 3600]");
    }
    return std::chrono::milliseconds(static_cast<int64_t>(std::ceil(s * 1000.0)));
  }

  void respond(const rclcpp::Service<SetBool>::SharedPtr & service, const std::shared_ptr<rmw_request_id_t> & header, bool success, const std::string & message){
    SetBool::Response response;
    response.success = success;
    response.message = message;
    if (command_events_) command_events_->finish_request(header, success, message);
    service->send_response(*header, response);
  }

  void lever(double z){
    std_msgs::msg::Float64 m;
    m.data = z;
    ascenseur_pub_->publish(m);
  }

  bool un_etage_tenu() const{
    return std::any_of(etages_.begin(), etages_.end(), [](const Etage & e) {return e.tenu;});
  }

  // Protocole du plugin : 1 <id> <PING|TAKE|RELEASE> <expiration_unix_ms>
  void envoyer(int i, const std::string & operation){
    const auto expiration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch() + timeout_).count();
    demande_->ids[i] = prefix_ + '-' + std::to_string(++counter_);
    std_msgs::msg::String msg;
    msg.data = "1 " + demande_->ids[i] + " " + operation + " " + std::to_string(expiration);
    etages_[i].commande->publish(msg);
  }

  void on_request(const rclcpp::Service<SetBool>::SharedPtr & service, std::shared_ptr<rmw_request_id_t> header, std::vector<int> etages, bool prendre){
    if (command_events_) command_events_->begin_request(header, prendre, etages.size() == 3 ? "all" : std::to_string(etages.front() + 1), !rotation_ && !demande_);
    if (rotation_) {
      respond(service, header, false, "Poignet en mouvement, réessayer après");
      return;
    }
    if (demande_) {
      respond(service, header, false, "Ventouses occupées");
      return;
    }
    const auto now = Clock::now();
    Demande d;
    d.service = service;
    d.header = std::move(header);
    d.etages = std::move(etages);
    d.prendre = prendre;
    d.echeance = now + timeout_ + std::chrono::seconds(1) + (prendre ? std::chrono::milliseconds(0) : delai_lacher_);
    d.prochain_ping = now + std::chrono::milliseconds(100);
    demande_ = std::move(d);
    // Le bridge peut être découvert avant le plugin : vérifier l'aller-retour avant d'agir.
    for (int i : demande_->etages) {envoyer(i, "PING");}
  }

  void on_result(int i, const std_msgs::msg::String & msg){
    std::istringstream input(msg.data);
    std::string version, id, status, code, extra;
    if (!(input >> version >> id >> status >> code) || version != "1" || (input >> extra) || !demande_ || id != demande_->ids[i]){
      return;
    }

    auto & d = *demande_;
    if (status == "OK" && code == "READY") {
      if (d.phase != Phase::PING) {return;}
      d.prets.insert(i);
      if (d.prets.size() < d.etages.size()) {return;}
      if (d.prendre) {
        d.phase = Phase::COMMANDE;
        for (int e : d.etages) {envoyer(e, "TAKE");}
      } else {
        d.phase = Phase::DESCENTE;   // poser avant de lâcher
        lever(0.0);
        d.fin_descente = Clock::now() + delai_lacher_;
      }
      return;
    }

    if (d.phase != Phase::COMMANDE || d.finis.count(i)) {return;}
    if (status == "OK" && (code == "TAKEN" || code == "ALREADY_HELD")) {
      etages_[i].tenu = true;
    } else if (status == "OK" && (code == "RELEASED" || code == "ALREADY_FREE")) {
      etages_[i].tenu = false;
    } else if (status == "ERROR") {
      d.erreur = "étage " + std::to_string(i + 1) + " : " + code;
    } else {
      return;   // résultat inconnu : ignoré
    }
    d.finis.insert(i);
    if (d.finis.size() == d.etages.size()) {terminer();}
  }

  void terminer(const std::string & erreur_forcee = ""){
    auto d = std::move(*demande_);
    demande_.reset();
    if (!erreur_forcee.empty()) {d.erreur = erreur_forcee;}
    // Ascenseur en haut dès qu'au moins une pierre est tenue, sinon en bas.
    if (un_etage_tenu()) {lever(hauteur_);}
    int tenus = 0;
    for (const auto & e : etages_) {tenus += e.tenu ? 1 : 0;}
    std::string message = d.prendre ? "Prise terminée" : "Lâcher terminé";
    message += ", " + std::to_string(tenus) + " étage(s) tenu(s)";
    if (!d.erreur.empty()) {message += " (" + d.erreur + ")";}
    respond(d.service, d.header, d.erreur.empty(), message);
  }

  void on_poignet(std::shared_ptr<rmw_request_id_t> header, bool debout){
    if (command_events_) command_events_->begin_request(header, debout, "wrist", !rotation_ && !demande_);
    if (rotation_) {
      respond(service_poignet_, header, false, "Poignet déjà en mouvement");
      return;
    }
    if (demande_) {
      respond(service_poignet_, header, false, "Ventouses en cours d'opération, réessayer après");
      return;
    }
    Rotation r;
    r.header = std::move(header);
    r.debout = debout;
    r.charge = un_etage_tenu();
    r.debut = Clock::now();
    if (r.charge) {
      lever(hauteur_rotation_);       // garde au sol pour la pile pendant la rotation
      r.debut += duree_montee_;
    }
    rotation_ = std::move(r);
  }

  void tick(){
    const auto now = Clock::now();
    
    if (rotation_ && !rotation_->lancee && now >= rotation_->debut) {
      std_msgs::msg::Float64 m;
      m.data = rotation_->debout ? angle_poignet_ : 0.0;
      poignet_pub_->publish(m);
      rotation_->lancee = true;
      rotation_->fin = now + duree_poignet_;
    }

    if (rotation_ && rotation_->lancee && now >= rotation_->fin) {
      if (rotation_->charge) {lever(hauteur_);}   // retour à la hauteur de transport
      respond(service_poignet_, rotation_->header, true,
        rotation_->debout ? "Poignet à 90°" : "Poignet à 0°");
      rotation_.reset();
    }

    if (!demande_) {return;}

    auto & d = *demande_;
    if (now >= d.echeance) {
      terminer(d.phase == Phase::PING ? "backend de ventouse indisponible" : "délai dépassé");
      return;
    }

    if (d.phase == Phase::PING && now >= d.prochain_ping) {
      for (int i : d.etages) {
        if (!d.prets.count(i)) {envoyer(i, "PING");}
      }
      d.prochain_ping = now + std::chrono::milliseconds(100);
    }

    if (d.phase == Phase::DESCENTE && now >= d.fin_descente) {
      d.phase = Phase::COMMANDE;
      for (int i : d.etages) {envoyer(i, "RELEASE");}
    }
  }

  std::unique_ptr<CommandEvents> command_events_;
  std::chrono::milliseconds timeout_{2000}, delai_lacher_{1200}, duree_poignet_{1500}, duree_montee_{400};
  double hauteur_{0.095}, hauteur_rotation_{0.12}, angle_poignet_{1.5708};
  std::string prefix_;
  uint64_t counter_{0};

  std::array<Etage, N> etages_;

  std::optional<Demande> demande_;
  std::optional<Rotation> rotation_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr ascenseur_pub_, poignet_pub_
  ;
  rclcpp::Service<SetBool>::SharedPtr tous_, service_poignet_;
  rclcpp::TimerBase::SharedPtr timer_;

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VentouseService>());
  rclcpp::shutdown();
  return 0;
}