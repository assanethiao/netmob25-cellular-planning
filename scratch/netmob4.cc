/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Simulation LTE - Paris (zone NetMob25)
 * 
 *
 * Zone : ~1 km² autour de 48.8918°N / 2.417°E
 *        
 *        Secteur La Défense
 *
 * 25 UE : 5 WALKING, 15 TRANSPORT, 5 DRIVING
 * 3 eNB en triangle, bande LTE 20 MHz (100 RB)
 *
 * Fichiers .txt générés par lteHelper->EnableTraces() :
 *   - DlRlcStats.txt      : stats RLC downlink
 *   - UlRlcStats.txt      : stats RLC uplink
 *   - DlMacStats.txt      : stats MAC downlink
 *   - UlMacStats.txt      : stats MAC uplink
 *   - DlPdcpStats.txt     : stats PDCP downlink
 *   - UlPdcpStats.txt     : stats PDCP uplink
 *   - DlRsrpSinrStats.txt : qualité signal (RSRP/SINR) par UE
 *
 * Fichier XML généré :
 *   - paris-flowmon.xml   : FlowMonitor (débit, latence, perte)
 *
 * Placement des 3 eNB (coordonnées relatives en mètres) :
 *   eNB 0 : (550, 800) 
 *   eNB 1 : (200, 200)     
 *   eNB 2 : (900, 200) 
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/netmob25-mobility-model.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include <iomanip>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ParisCentreLteSimulation");

// ─────────────────────────────────────────────────────────────────
// Variables globales pour la fonction d'affichage des positions
// (nécessaire car Simulator::Schedule ne supporte pas les lambdas)
// ─────────────────────────────────────────────────────────────────
static NodeContainer g_allUeNodes;
static uint32_t      g_nWalking   = 0;
static uint32_t      g_nTransport = 0;
static uint32_t      g_totalUe    = 0;

void
PrintUePositions (void)
{
  std::cout << "\n[t=" << Simulator::Now ().GetSeconds () << "s] Positions UE :" << std::endl;
  for (uint32_t i = 0; i < g_totalUe; i++)
    {
      Ptr<MobilityModel> mob = g_allUeNodes.Get (i)->GetObject<MobilityModel> ();
      if (!mob) continue;
      Vector pos   = mob->GetPosition ();
      Vector vel   = mob->GetVelocity ();
      std::string mode;
      if      (i < g_nWalking)                     mode = "WALK ";
      else if (i < g_nWalking + g_nTransport)      mode = "TRANS";
      else                                          mode = "DRIVE";
      double speed = std::sqrt (vel.x*vel.x + vel.y*vel.y);
      std::cout << "  UE" << std::setw(2) << i
                << " [" << mode << "]"
                << "  pos=(" << std::fixed << std::setprecision(1)
                << pos.x << ", " << pos.y << ")"
                << "  v=" << std::setprecision(2) << speed << " m/s"
                << std::endl;
    }
}

// ─────────────────────────────────────────────────────────────────
// Paramètres de simulation
// ─────────────────────────────────────────────────────────────────

// Zone Paris centre cohérente avec le modèle NetMob25
// refLat=48.852737 / refLon=2.350699 → ~1 km x 1 km
static const double AREA_X = 1100.0;  // mètres
static const double AREA_Y = 1000.0;  // mètres

// UE par mode (TransportMode du modèle NetMob25)
static const uint32_t N_WALKING   = 5;
static const uint32_t N_TRANSPORT = 15;  // TRANSPORT = bus/métro/RER
static const uint32_t N_DRIVING   = 5;   // DRIVING   = voiture
static const uint32_t N_UE        = N_WALKING + N_TRANSPORT + N_DRIVING; // = 25

static const uint32_t N_ENB       = 3;
static const double   SIM_TIME    = 5.0; // secondes

// Paramètres trafic UDP
// Intervalle 20ms → ~50 paquets/s par UE → trafic suffisant pour remplir les .txt
static const uint32_t PKT_SIZE     = 1024;           // octets
static const uint32_t PKT_INTERVAL = 10;             // millisecondes
static const uint32_t MAX_PACKETS  = 3000;           // par UE (60s x 50 pkt/s)
static const uint16_t DL_PORT      = 1234;
static const uint16_t UL_PORT      = 5678;

// ─────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────

int
main (int argc, char *argv[])
{
  uint32_t nWalking   = N_WALKING;
  uint32_t nTransport = N_TRANSPORT;
  uint32_t nDriving   = N_DRIVING;
  double   simTime    = SIM_TIME;

  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (46.0));
  CommandLine cmd (__FILE__);
  cmd.AddValue ("nWalking",   "Nombre de pietons",           nWalking);
  cmd.AddValue ("nTransport", "Nombre transports en commun", nTransport);
  cmd.AddValue ("nDriving",   "Nombre de voitures",          nDriving);
  cmd.AddValue ("simTime",    "Duree simulation (secondes)", simTime);
  cmd.Parse (argc, argv);

  uint32_t totalUe = nWalking + nTransport + nDriving;

  // ── Logs ──────────────────────────────────────────────────────
  LogComponentEnable ("Netmob25MobilityModel", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpClient",             LOG_LEVEL_INFO);
  LogComponentEnable ("UdpServer",             LOG_LEVEL_INFO);
  LogComponentEnable ("PacketSink",            LOG_LEVEL_INFO);

  std::cout << "================================================" << std::endl;
  std::cout << "  Simulation LTE - Paris (NetMob25)"       << std::endl;
  std::cout << ""                                                 << std::endl;
  std::cout << "================================================" << std::endl;
  std::cout << "  Zone          : " << AREA_X << "m x " << AREA_Y << "m" << std::endl;
  std::cout << "  Total UE      : " << totalUe    << std::endl;
  std::cout << "    WALKING     : " << nWalking   << std::endl;
  std::cout << "    TRANSPORT   : " << nTransport << std::endl;
  std::cout << "    DRIVING     : " << nDriving   << std::endl;
  std::cout << "  Stations eNB  : " << N_ENB      << std::endl;
  std::cout << "  Bande LTE     : 20 MHz (100 RB, ~150 Mbps/cell)" << std::endl;
  std::cout << "  Duree         : " << simTime    << "s"           << std::endl;
  std::cout << "  Trafic UDP    : " << PKT_SIZE << "B / " << PKT_INTERVAL << "ms par UE" << std::endl;
  std::cout << "================================================" << std::endl;

  // ── LTE + EPC ─────────────────────────────────────────────────
  Ptr<LteHelper>             lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // Bande 20 MHz = 100 Resource Blocks
  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (100));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (100));

  // Scheduler Round Robin (équitable entre UE)
  lteHelper->SetSchedulerType ("ns3::RrFfMacScheduler");

  // Modèle de propagation urbain Paris (Okumura-Hata, valeurs par défaut : Urban/Large)
 /* lteHelper->SetPathlossModelType (
    TypeId::LookupByName ("ns3::OkumuraHataPropagationLossModel"));


// Hauteur eNB : 30m (déjà dans enbPos), hauteur UE : 1.5m
Config::SetDefault ("ns3::OkumuraHataPropagationLossModel::Environment",
                    EnumValue (OkumuraHataPropagationLossModel::UrbanEnvironment));
Config::SetDefault ("ns3::OkumuraHataPropagationLossModel::CitySize",
                    EnumValue (OkumuraHataPropagationLossModel::LargeCity));
*/

  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisPropagationLossModel"));
// ── Nœuds ─────────────────────────────────────────────────────
  NodeContainer enbNodes;
  enbNodes.Create (N_ENB);

  NodeContainer ueWalking, ueTransport, ueDriving;
  ueWalking.Create   (nWalking);
  ueTransport.Create (nTransport);
  ueDriving.Create   (nDriving);

  NodeContainer allUeNodes;
  allUeNodes.Add (ueWalking);
  allUeNodes.Add (ueTransport);
  allUeNodes.Add (ueDriving);

  // Initialiser les variables globales pour PrintUePositions
  g_allUeNodes  = allUeNodes;
  g_nWalking    = nWalking;
  g_nTransport  = nTransport;
  g_totalUe     = totalUe;

  // ── Mobilité eNB (positions fixes) ────────────────────────────
  //
  //  Les coordonnées sont relatives au point de référence 
  //
  MobilityHelper enbMobility;
  Ptr<ListPositionAllocator> enbPos = CreateObject<ListPositionAllocator> ();
  enbPos->Add (Vector (550.0, 800.0, 30.0)); // eNB 0 
  enbPos->Add (Vector (200.0, 200.0, 30.0)); // eNB 1 
  enbPos->Add (Vector (900.0, 200.0, 30.0)); // eNB 2 
  enbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  enbMobility.SetPositionAllocator (enbPos);
  enbMobility.Install (enbNodes);


  // ── Mobilité UE via NetMob25 ───────────────────────────────────

  // --- WALKING ---
  MobilityHelper mobWalking;
  mobWalking.SetMobilityModel (
    "ns3::Netmob25MobilityModel",
    "StartTime",      TimeValue    (Seconds (0.0)),
    "UpdateInterval", TimeValue    (Seconds (2.0)),
    "ModelPath",      StringValue  ("model.pt"),
    "TransportMode",  StringValue  ("WALKING"),
    "TripLength",     UintegerValue (200));
  mobWalking.Install (ueWalking);

  // --- TRANSPORT (bus / métro / RER) ---
  MobilityHelper mobTransport;
  mobTransport.SetMobilityModel (
    "ns3::Netmob25MobilityModel",
    "StartTime",      TimeValue    (Seconds (0.0)),
    "UpdateInterval", TimeValue    (Seconds (2.0)),
    "ModelPath",      StringValue  ("model.pt"),
    "TransportMode",  StringValue  ("TRANSPORT"),
    "TripLength",     UintegerValue (500));
  mobTransport.Install (ueTransport);

  // --- CAR (voiture) ---
  MobilityHelper mobDriving;
  mobDriving.SetMobilityModel (
    "ns3::Netmob25MobilityModel",
    "StartTime",      TimeValue    (Seconds (0.0)),
    "UpdateInterval", TimeValue    (Seconds (2.0)),
    "ModelPath",      StringValue  ("model.pt"),
    "TransportMode",  StringValue  ("DRIVING"),
    "TripLength",     UintegerValue (400));
  mobDriving.Install (ueDriving);

  // ── Stack Internet sur les UE ──────────────────────────────────
  InternetStackHelper internet;
  internet.Install (allUeNodes);

  // ── Devices LTE ───────────────────────────────────────────────
  NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice  (enbNodes);
  NetDeviceContainer ueDevs  = lteHelper->InstallUeDevice   (allUeNodes);

  // Adresses IP pour les UE
  epcHelper->AssignUeIpv4Address (ueDevs);

  // Route par défaut pour les UE vers le PGW
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  for (uint32_t i = 0; i < totalUe; i++)
    {
      Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting (allUeNodes.Get (i)->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

  // ── Attachement UE → eNB (meilleur signal) ────────────────────
  lteHelper->Attach (ueDevs);

  // ── Activer les traces AVANT le trafic ────────────────────────
  //
  // IMPORTANT : EnableTraces() doit être appelé AVANT Simulator::Run()
  // Il génère automatiquement :
  //   DlRlcStats.txt / UlRlcStats.txt
  //   DlMacStats.txt / UlMacStats.txt
  //   DlPdcpStats.txt / UlPdcpStats.txt
  //   DlRsrpSinrStats.txt
  //
  lteHelper->EnableTraces ();

  // ── FlowMonitor ────────────────────────────────────────────────
  FlowMonitorHelper flowMonHelper;
  Ptr<FlowMonitor>  flowMon = flowMonHelper.InstallAll ();

  // ── Applications : trafic DL + UL ─────────────────────────────
  //
  // Pour garantir des données dans les fichiers .txt, on installe :
  //   1) Un UdpServer sur chaque UE (reçoit le trafic DL depuis le RemoteHost)
  //   2) Un PacketSink sur le RemoteHost (reçoit le trafic UL depuis les UE)
  //   3) Un OnOffApplication DL : RemoteHost → chaque UE
  //   4) Un OnOffApplication UL : chaque UE  → RemoteHost
  //
  // OnOff avec dataRate 500Kbps et pktSize 1024B → flux continu et réaliste
  //

  // Nœud RemoteHost côté réseau (au-delà du PGW)
  Ptr<Node> remoteHost;
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  remoteHost = remoteHostContainer.Get (0);
  internet.Install (remoteHostContainer);

  // Lien P2P entre PGW et RemoteHost
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute  ("DataRate", DataRateValue (DataRate ("100Gb/s")));
  p2ph.SetDeviceAttribute  ("Mtu",      UintegerValue (1500));
  p2ph.SetChannelAttribute ("Delay",    TimeValue (MilliSeconds (1)));

  NetDeviceContainer internetDevices = p2ph.Install (epcHelper->GetPgwNode (), remoteHost);

  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

  // Route vers les UE depuis le RemoteHost
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
    ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (
    Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"),
    internetIpIfaces.GetAddress (0), 1);

  std::cout << "\nInstallation des applications (DL + UL)..." << std::endl;

  for (uint32_t i = 0; i < totalUe; i++)
    {
      // Adresse IP de cet UE
      Ptr<Ipv4> ueIpv4  = allUeNodes.Get (i)->GetObject<Ipv4> ();
      Ipv4Address ueAddr = ueIpv4->GetAddress (1, 0).GetLocal ();

      // ── Trafic Downlink : RemoteHost → UE ──────────────────────
      // Serveur UDP sur l'UE
      UdpServerHelper dlServer (DL_PORT);
      ApplicationContainer dlServerApp = dlServer.Install (allUeNodes.Get (i));
      dlServerApp.Start (Seconds (0.5));
      dlServerApp.Stop  (Seconds (simTime));

      // Client UDP sur le RemoteHost → envoie vers l'UE
      UdpClientHelper dlClient (ueAddr, DL_PORT);
      dlClient.SetAttribute ("Interval",   TimeValue   (MilliSeconds (PKT_INTERVAL)));
      dlClient.SetAttribute ("MaxPackets", UintegerValue (MAX_PACKETS));
      dlClient.SetAttribute ("PacketSize", UintegerValue (PKT_SIZE));
      ApplicationContainer dlClientApp = dlClient.Install (remoteHost);
      dlClientApp.Start (Seconds (1.0));
      dlClientApp.Stop  (Seconds (simTime));

      // ── Trafic Uplink : UE → RemoteHost ────────────────────────
      // Sink sur le RemoteHost
      PacketSinkHelper ulSink ("ns3::UdpSocketFactory",
                               InetSocketAddress (remoteHostAddr, UL_PORT + i));
      ApplicationContainer ulSinkApp = ulSink.Install (remoteHost);
      ulSinkApp.Start (Seconds (0.5));
      ulSinkApp.Stop  (Seconds (simTime));

      // Client UDP sur l'UE → envoie vers le RemoteHost
      UdpClientHelper ulClient (remoteHostAddr, UL_PORT + i);
      ulClient.SetAttribute ("Interval",   TimeValue   (MilliSeconds (PKT_INTERVAL)));
      ulClient.SetAttribute ("MaxPackets", UintegerValue (MAX_PACKETS));
      ulClient.SetAttribute ("PacketSize", UintegerValue (PKT_SIZE));
      ApplicationContainer ulClientApp = ulClient.Install (allUeNodes.Get (i));
      ulClientApp.Start (Seconds (1.0));
      ulClientApp.Stop  (Seconds (simTime));
    }

  std::cout << "  " << totalUe << " flux DL + " << totalUe << " flux UL installes" << std::endl;
  std::cout << "  Debit par UE : ~" << (PKT_SIZE * 8 * 1000 / PKT_INTERVAL) / 1000
            << " Kbps (DL) + ~" << (PKT_SIZE * 8 * 1000 / PKT_INTERVAL) / 1000
            << " Kbps (UL)" << std::endl;

  // ── Affichage positions toutes les 10 secondes ────────────────
  for (double t = 10.0; t <= simTime; t += 10.0)
    Simulator::Schedule (Seconds (t), &PrintUePositions);

  // ── Lancement ─────────────────────────────────────────────────
  std::cout << "\nDemarrage simulation (" << simTime << "s)..." << std::endl;
  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();

  // ── Export FlowMonitor XML ─────────────────────────────────────
  flowMon->SerializeToXmlFile ("paris-flowmon.xml", true, true);

  // ── Résumé console post-simulation ────────────────────────────
  std::cout << "\n================================================" << std::endl;
  std::cout << "  Resultats de la simulation"                       << std::endl;
  std::cout << "================================================" << std::endl;

  flowMon->CheckForLostPackets ();
  std::map<FlowId, FlowMonitor::FlowStats> stats = flowMon->GetFlowStats ();

  double sumTput  = 0.0;
  double sumDelay = 0.0;
  double sumLoss  = 0.0;
  uint32_t nFlow  = 0;

  for (auto &kv : stats)
    {
      FlowMonitor::FlowStats &fs = kv.second;
      if (fs.rxPackets == 0) continue;
      double tput  = fs.rxBytes * 8.0 / simTime / 1e6;
      double delay = fs.delaySum.GetMilliSeconds () / fs.rxPackets;
      double loss  = 100.0 * fs.lostPackets / (fs.rxPackets + fs.lostPackets + 1);
      sumTput  += tput;
      sumDelay += delay;
      sumLoss  += loss;
      nFlow++;
    }

  if (nFlow > 0)
    {
      std::cout << "  Flux actifs           : " << nFlow << std::endl;
      std::cout << "  Debit moyen / flux    : " << sumTput  / nFlow << " Mbps" << std::endl;
      std::cout << "  Latence moyenne       : " << sumDelay / nFlow << " ms"   << std::endl;
      std::cout << "  Perte paquets moy.    : " << sumLoss  / nFlow << " %"    << std::endl;
      std::cout << "  Debit total reseau    : " << sumTput          << " Mbps" << std::endl;
    }

  std::cout << "\nFichiers generes :" << std::endl;
  std::cout << "  paris-flowmon.xml    → latence, debit, perte par flux" << std::endl;
  std::cout << "  DlRlcStats.txt       → RLC downlink par UE"            << std::endl;
  std::cout << "  UlRlcStats.txt       → RLC uplink par UE"              << std::endl;
  std::cout << "  DlMacStats.txt       → MAC downlink"                   << std::endl;
  std::cout << "  UlMacStats.txt       → MAC uplink"                     << std::endl;
  std::cout << "  DlPdcpStats.txt      → PDCP downlink"                  << std::endl;
  std::cout << "  UlPdcpStats.txt      → PDCP uplink"                    << std::endl;
  std::cout << "  DlRsrpSinrStats.txt  → qualite signal par UE"          << std::endl;
  std::cout << "================================================" << std::endl;

  Simulator::Destroy ();
  std::cout << "\nSimulation terminee avec succes !" << std::endl;
  return 0;
}
