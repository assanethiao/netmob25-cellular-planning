/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * ============================================================
 *  Netmob25 LTE Simulation - Version 04 (Optimisée)
 * ============================================================
 *  Auteur    : Simulation NS-3.44
 *  Objectif  : 50 UE + 5 eNodeB, mobilité Netmob25 WALKING
 *
 *  Fonctionnalités :
 *    - CSV : temps, position (x,y), débit reçu par UE (Mbps)
 *    - Placement eNodeB optimisé (grille + centroïde)
 *    - Attachement UE → eNodeB par distance minimale
 *    - Trafic UDP réaliste (OnOff ~5 Mbps/UE)
 *    - NetAnim avec noms et couleurs
 *    - Zone d'étude délimitée automatiquement
 * ============================================================
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/netmob25-mobility-model.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/node-list.h"
#include <fstream>
#include <map>
#include <cmath>
#include <vector>
#include <limits>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Netmob25Optimizedv4");

// ============================================================
//  STRUCTURES DE DONNÉES
// ============================================================

// Statistiques de débit par UE (pour calcul différentiel)
struct UeThroughputStats
{
  uint64_t prevRxBytes = 0;
  double   prevTime    = 0.0;
  double   currentThroughputMbps = 0.0;
};

// Variables globales partagées entre les callbacks
std::map<uint32_t, UeThroughputStats> g_ueStats;   // nodeId -> stats
std::ofstream g_csvFile;                             // fichier CSV de sortie

// ============================================================
//  UTILITAIRES GÉOMÉTRIQUES
// ============================================================

/**
 * Calcule la distance euclidienne 2D entre deux vecteurs.
 */
double Distance2D (const Vector& a, const Vector& b)
{
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  return std::sqrt (dx * dx + dy * dy);
}

/**
 * Retourne l'index du eNodeB le plus proche d'un UE.
 *
 * @param uePos     Position du UE
 * @param enbNodes  Conteneur des nœuds eNodeB
 * @return          Index du eNodeB le plus proche
 */
uint32_t FindNearestEnb (const Vector& uePos, const NodeContainer& enbNodes)
{
  uint32_t bestIdx  = 0;
  double   bestDist = std::numeric_limits<double>::max ();

  for (uint32_t i = 0; i < enbNodes.GetN (); ++i)
    {
      Ptr<MobilityModel> m = enbNodes.Get (i)->GetObject<MobilityModel> ();
      if (!m) continue;
      double d = Distance2D (uePos, m->GetPosition ());
      if (d < bestDist)
        {
          bestDist = d;
          bestIdx  = i;
        }
    }
  return bestIdx;
}

// ============================================================
//  PLACEMENT DES eNodeB — GRILLE ADAPTÉE + CENTROÏDE
// ============================================================

/**
 * Calcule les positions optimales pour nEnb eNodeB dans une zone
 * rectangulaire [xMin, xMax] x [yMin, yMax].
 *
 * Stratégie :
 *   - Pour nEnb = 5 : 4 coins (avec marge 15%) + 1 centroïde
 *   - Pour nEnb quelconque : grille NxM couvrant la zone uniformément
 *
 * Marge de 15% par rapport aux bords : les antennes réelles ne sont
 * jamais placées exactement sur les limites de la zone d'étude.
 *
 * @param xMin, xMax, yMin, yMax  Bornes de la zone d'étude (m)
 * @param nEnb                    Nombre d'eNodeB à placer
 * @return                        Vecteur de positions Vector(x, y, hauteur)
 */
std::vector<Vector> ComputeEnbPositions (double xMin, double xMax,
                                         double yMin, double yMax,
                                         uint32_t nEnb)
{
  std::vector<Vector> positions;
  double marginX = 0.15 * (xMax - xMin);
  double marginY = 0.15 * (yMax - yMin);
  double enbHeight = 30.0; // Hauteur typique antenne LTE (m)

  double x0 = xMin + marginX;
  double x1 = xMax - marginX;
  double y0 = yMin + marginY;
  double y1 = yMax - marginY;
  double cx = (xMin + xMax) / 2.0;
  double cy = (yMin + yMax) / 2.0;

  if (nEnb == 1)
    {
      // Un seul eNodeB : centroïde
      positions.push_back (Vector (cx, cy, enbHeight));
    }
  else if (nEnb == 4)
    {
      // 4 coins
      positions.push_back (Vector (x0, y0, enbHeight));
      positions.push_back (Vector (x1, y0, enbHeight));
      positions.push_back (Vector (x0, y1, enbHeight));
      positions.push_back (Vector (x1, y1, enbHeight));
    }
  else if (nEnb == 5)
    {
      // 4 coins + centroïde (configuration optimale pour la couverture)
      positions.push_back (Vector (x0, y0, enbHeight));
      positions.push_back (Vector (x1, y0, enbHeight));
      positions.push_back (Vector (x0, y1, enbHeight));
      positions.push_back (Vector (x1, y1, enbHeight));
      positions.push_back (Vector (cx, cy, enbHeight)); // centroïde central
    }
  else
    {
      // Grille uniforme NxM pour tout autre nombre d'eNodeB
      // Cherche les facteurs les plus proches de sqrt(nEnb)
      uint32_t cols = (uint32_t) std::ceil (std::sqrt ((double) nEnb));
      uint32_t rows = (uint32_t) std::ceil ((double) nEnb / cols);

      double stepX = (x1 - x0) / std::max (1u, cols - 1);
      double stepY = (y1 - y0) / std::max (1u, rows - 1);

      uint32_t placed = 0;
      for (uint32_t r = 0; r < rows && placed < nEnb; ++r)
        {
          for (uint32_t c = 0; c < cols && placed < nEnb; ++c)
            {
              double px = (cols == 1) ? cx : x0 + c * stepX;
              double py = (rows == 1) ? cy : y0 + r * stepY;
              positions.push_back (Vector (px, py, enbHeight));
              ++placed;
            }
        }
    }

  return positions;
}

// ============================================================
//  MISE À JOUR DU CSV : position + débit
// ============================================================

/**
 * Callback planifié toutes les secondes.
 * Lit la position de chaque UE et son débit instantané,
 * puis écrit une ligne dans le CSV.
 *
 * Le débit est mis à jour par UpdateThroughput() juste avant.
 *
 * @param ueNodes  Conteneur UE
 * @param nNodes   Nombre de nœuds UE
 */
void WriteCsvSnapshot (const NodeContainer& ueNodes, uint32_t nNodes)
{
  double t = Simulator::Now ().GetSeconds ();

  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<MobilityModel> mob = ueNodes.Get (i)->GetObject<MobilityModel> ();
      if (!mob) continue;

      Vector pos = mob->GetPosition ();
      uint32_t nodeId = ueNodes.Get (i)->GetId ();

      double throughput = 0.0;
      auto it = g_ueStats.find (nodeId);
      if (it != g_ueStats.end ())
        {
          throughput = it->second.currentThroughputMbps;
        }

      // Format CSV : temps,noeud,x,y,throughput_Mbps
      g_csvFile << t << "," << i << "," << pos.x << "," << pos.y
                << "," << throughput << "\n";
    }
}

// ============================================================
//  MISE À JOUR DU DÉBIT via FlowMonitor
// ============================================================

/**
 * Appelé périodiquement pour mettre à jour les débits par UE.
 *
 * Principe : FlowMonitor accumule les octets reçus par flux.
 * On calcule le débit différentiel :
 *   Tput(Mbps) = (ΔBytes × 8) / (Δt × 1e6)
 *
 * Chaque flux est associé à un UE via son adresse IP destination
 * (ueIpAddrs). On fait la somme des flux entrants vers chaque UE.
 *
 * @param monitor        Pointeur FlowMonitor
 * @param classifier     Classifieur IP pour retrouver les adresses
 * @param ueIpAddrs      Tableau IP des UE
 * @param nNodes         Nombre de UE
 * @param ueNodes        Conteneur UE (pour nodeId)
 * @param interval       Intervalle de mesure (secondes)
 */
void UpdateThroughput (Ptr<FlowMonitor>              monitor,
                       Ptr<Ipv4FlowClassifier>       classifier,
                       const Ipv4InterfaceContainer& ueIpAddrs,
                       uint32_t                      nNodes,
                       const NodeContainer&          ueNodes,
                       double                        interval)
{
  monitor->CheckForLostPackets ();
  FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();

  // Map : adresse IP destination → octets reçus cumulés (tous flux confondus)
  std::map<Ipv4Address, uint64_t> rxBytesPerDst;

  for (auto& kv : stats)
    {
      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (kv.first);
      rxBytesPerDst[t.destinationAddress] += kv.second.rxBytes;
    }

  double now = Simulator::Now ().GetSeconds ();

  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ipv4Address addr = ueIpAddrs.GetAddress (i);
      uint32_t nodeId  = ueNodes.Get (i)->GetId ();

      uint64_t rxBytes = 0;
      auto it = rxBytesPerDst.find (addr);
      if (it != rxBytesPerDst.end ())
        rxBytes = it->second;

      UeThroughputStats& s = g_ueStats[nodeId];

      double deltaBytes = (double)(rxBytes - s.prevRxBytes);
      double deltaTime  = now - s.prevTime;

      if (deltaTime > 0.0)
        s.currentThroughputMbps = (deltaBytes * 8.0) / (deltaTime * 1e6);

      s.prevRxBytes = rxBytes;
      s.prevTime    = now;
    }
}

/**
 * Fonction maître planifiée : met à jour les débits PUIS écrit le CSV.
 * L'ordre est important : throughput d'abord, CSV ensuite.
 */
void PeriodicUpdate (Ptr<FlowMonitor>              monitor,
                     Ptr<Ipv4FlowClassifier>       classifier,
                     const Ipv4InterfaceContainer& ueIpAddrs,
                     uint32_t                      nNodes,
                     const NodeContainer&          ueNodes,
                     double                        interval,
                     double                        simTime)
{
  // 1. Mettre à jour les débits
  UpdateThroughput (monitor, classifier, ueIpAddrs, nNodes, ueNodes, interval);

  // 2. Écrire snapshot CSV
  WriteCsvSnapshot (ueNodes, nNodes);

  // 3. Afficher console (positions uniquement, pour ne pas surcharger)
  double t = Simulator::Now ().GetSeconds ();
  std::cout << "[" << t << "s] Snapshot écrit (" << nNodes << " UE)" << std::endl;

  // 4. Re-planifier si la simulation continue
  if (t + interval <= simTime)
    {
      Simulator::Schedule (Seconds (interval), &PeriodicUpdate,
                           monitor, classifier, ueIpAddrs,
                           nNodes, ueNodes, interval, simTime);
    }
}

// ============================================================
//  MAIN
// ============================================================

int main (int argc, char *argv[])
{
  // ----------------------------------------------------------
  //  Paramètres de simulation
  // ----------------------------------------------------------
  uint32_t nNodes   = 50;       // Nombre de UE
  uint32_t nENodeB  = 5;        // Nombre d'eNodeB
  double   simTime  = 100.0;    // Durée de simulation (s)
  double   interval = 1.0;      // Intervalle de mesure CSV (s)
  double   ueDataRateMbps = 5.0; // Débit cible par UE (Mbps)

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nNodes",        "Nombre de UE",              nNodes);
  cmd.AddValue ("nENodeB",       "Nombre d'eNodeB",           nENodeB);
  cmd.AddValue ("simTime",       "Durée simulation (s)",      simTime);
  cmd.AddValue ("ueDataRate",    "Débit cible UE (Mbps)",     ueDataRateMbps);
  cmd.Parse (argc, argv);

  // ----------------------------------------------------------
  //  Logging
  // ----------------------------------------------------------
  LogComponentEnable ("Netmob25MobilityModel", LOG_LEVEL_INFO);
  // Décommenter pour debug trafic :
  // LogComponentEnable ("OnOffApplication",      LOG_LEVEL_INFO);
  // LogComponentEnable ("PacketSink",            LOG_LEVEL_INFO);

  std::cout << "======================================================" << std::endl;
  std::cout << "  Netmob25 LTE Simulation v04 - Optimisée"             << std::endl;
  std::cout << "  UE       : " << nNodes                               << std::endl;
  std::cout << "  eNodeB   : " << nENodeB                              << std::endl;
  std::cout << "  Durée    : " << simTime << " s"                      << std::endl;
  std::cout << "  Débit/UE : " << ueDataRateMbps << " Mbps (cible)"   << std::endl;
  std::cout << "======================================================" << std::endl;

  // ----------------------------------------------------------
  //  Fichier CSV de sortie
  // ----------------------------------------------------------
  g_csvFile.open ("trajectories_v04_50UE_5eNode.csv");
  if (!g_csvFile.is_open ())
    {
      NS_FATAL_ERROR ("Impossible d'ouvrir le fichier CSV de sortie.");
    }
  // En-tête CSV
  g_csvFile << "time_s,ue_index,x_m,y_m,throughput_Mbps\n";

  // ----------------------------------------------------------
  //  Infrastructure LTE : LteHelper + EPC
  // ----------------------------------------------------------
  Ptr<LteHelper>             lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // Modèle de propagation extérieur urbain.
  // NOTE : OkumuraHata impose hb > 0 ET hm > 0 — les UE Netmob25 ont z=0
  // par défaut, ce qui provoque un crash (NS_ASSERT).
  // On utilise Cost231 qui tolère z=0 côté mobile, avec une hauteur eNodeB
  // fixée à 30 m dans les positions. Si tu veux OkumuraHata, il faudrait
  // forcer z=1.5 sur tous les UE via un allocateur de position initial.
  lteHelper->SetAttribute ("PathlossModel",
                           StringValue ("ns3::Cost231PropagationLossModel"));

  // Planificateur de ressources radio
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  // ----------------------------------------------------------
  //  Création des nœuds
  // ----------------------------------------------------------
  NodeContainer enbNodes, ueNodes;
  enbNodes.Create (nENodeB);
  ueNodes.Create (nNodes);

  // ----------------------------------------------------------
  //  MOBILITÉ UE — Netmob25 (WALKING)
  // ----------------------------------------------------------
  MobilityHelper ueMobility;
  ueMobility.SetMobilityModel (
    "ns3::Netmob25MobilityModel",
    "StartTime",      TimeValue (Seconds (0.0)),
    "UpdateInterval", TimeValue (Seconds (2.0)),
    "ModelPath",      StringValue ("model.pt"),
    "TransportMode",  StringValue ("WALKING"),
    "TripLength",     UintegerValue (100));
  ueMobility.Install (ueNodes);

  // ----------------------------------------------------------
  //  DÉLIMITATION DE LA ZONE D'ÉTUDE
  //  On lit les positions initiales pour définir les bornes.
  //  Ces bornes serviront au placement des eNodeB.
  // ----------------------------------------------------------
  double xMin =  1e9, xMax = -1e9;
  double yMin =  1e9, yMax = -1e9;

  std::cout << "\n--- Positions initiales des UE ---" << std::endl;
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<MobilityModel> mob = ueNodes.Get (i)->GetObject<MobilityModel> ();
      if (!mob) continue;
      Vector pos = mob->GetPosition ();
      if (pos.x < xMin) xMin = pos.x;
      if (pos.x > xMax) xMax = pos.x;
      if (pos.y < yMin) yMin = pos.y;
      if (pos.y > yMax) yMax = pos.y;
    }

  std::cout << "Zone d'étude délimitée :" << std::endl;
  std::cout << "  X : [" << xMin << ", " << xMax << "] m"
            << "  (largeur = " << (xMax - xMin) << " m)" << std::endl;
  std::cout << "  Y : [" << yMin << ", " << yMax << "] m"
            << "  (hauteur = " << (yMax - yMin) << " m)" << std::endl;

  // ----------------------------------------------------------
  //  PLACEMENT OPTIMAL DES eNodeB
  //  Stratégie : 4 coins (marge 15%) + 1 centroïde
  // ----------------------------------------------------------
  std::vector<Vector> enbPositions =
    ComputeEnbPositions (xMin, xMax, yMin, yMax, nENodeB);

  std::cout << "\n--- Positions des eNodeB (optimisées) ---" << std::endl;
  Ptr<ListPositionAllocator> enbPosAlloc = CreateObject<ListPositionAllocator> ();
  for (uint32_t i = 0; i < enbPositions.size (); ++i)
    {
      enbPosAlloc->Add (enbPositions[i]);
      std::cout << "  eNodeB-" << i
                << " : (" << enbPositions[i].x
                << ", "   << enbPositions[i].y
                << ", "   << enbPositions[i].z << ")" << std::endl;
    }

  MobilityHelper enbMobility;
  enbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  enbMobility.SetPositionAllocator (enbPosAlloc);
  enbMobility.Install (enbNodes);

  // ----------------------------------------------------------
  //  INSTALLATION DES DISPOSITIFS LTE
  // ----------------------------------------------------------
  NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueDevs  = lteHelper->InstallUeDevice  (ueNodes);

  // ----------------------------------------------------------
  //  PILE INTERNET SUR LES UE
  // ----------------------------------------------------------
  InternetStackHelper internet;
  internet.Install (ueNodes);

  Ipv4InterfaceContainer ueIpAddrs = epcHelper->AssignUeIpv4Address (ueDevs);

  // ----------------------------------------------------------
  //  ATTACHEMENT UE → eNodeB PAR DISTANCE MINIMALE
  //
  //  Pour chaque UE, on cherche le eNodeB le plus proche de sa
  //  position initiale. C'est une approximation valide au démarrage ;
  //  une simulation avancée utiliserait un handover dynamique.
  // ----------------------------------------------------------
  std::cout << "\n--- Attachement UE → eNodeB (par distance) ---" << std::endl;
  for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
      Ptr<MobilityModel> ueMob = ueNodes.Get (i)->GetObject<MobilityModel> ();
      Vector uePos = ueMob ? ueMob->GetPosition () : Vector (0, 0, 0);

      uint32_t bestEnb = FindNearestEnb (uePos, enbNodes);
      lteHelper->Attach (ueDevs.Get (i), enbDevs.Get (bestEnb));

      if (i < 10) // Afficher les 10 premiers pour vérification
        std::cout << "  UE-" << i << " → eNodeB-" << bestEnb
                  << " (dist=" << (int)Distance2D(uePos, enbPositions[bestEnb]) << "m)" << std::endl;
    }

  // ----------------------------------------------------------
  //  APPLICATION DE TRAFIC RÉALISTE (UDP OnOff → PacketSink)
  //
  //  Modèle choisi : OnOffApplication
  //    - DataRate  : ~5 Mbps par UE (simuler une session vidéo/data 5G)
  //    - PacketSize: 1400 bytes (proche MTU)
  //    - OnTime    : 100% (flux continu)
  //    - Direction : un serveur PacketSink sur chaque UE,
  //                  un client OnOff sur le nœud RemoteHost (via EPC)
  //
  //  Architecture : RemoteHost (EPC) → [Internet] → PGW → eNodeB → UE
  //  Les UE reçoivent le trafic descendant (downlink) depuis le serveur distant.
  // ----------------------------------------------------------

  Ptr<Node> pgw = epcHelper->GetPgwNode ();

  // Nœud distant (serveur de trafic)
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  internet.Install (remoteHostContainer);

  // Lien point-à-point entre RemoteHost et PGW
  PointToPointHelper p2p;
  p2p.SetDeviceAttribute  ("DataRate", StringValue ("10Gbps"));
  p2p.SetChannelAttribute ("Delay",    StringValue ("5ms"));
  NetDeviceContainer internetDevices = p2p.Install (pgw, remoteHost);

  // Adresses IP sur le lien PGW ↔ RemoteHost
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("1.0.0.0", "255.255.255.0");
  Ipv4InterfaceContainer internetIfaces = ipv4.Assign (internetDevices);

  // Route par défaut sur le RemoteHost vers le réseau UE
  Ipv4StaticRoutingHelper routingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
    routingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (
    Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.255.255.0"), 1);

  // Débit cible configuré via OnOff
  std::string dataRateStr =
    std::to_string ((uint32_t)(ueDataRateMbps * 1e6)) + "bps";

  uint16_t dlPort = 1234;

  for (uint32_t i = 0; i < nNodes; ++i)
    {
      // --- Serveur (PacketSink) sur le UE ---
      PacketSinkHelper sink ("ns3::UdpSocketFactory",
                             InetSocketAddress (Ipv4Address::GetAny (), dlPort));
      ApplicationContainer sinkApp = sink.Install (ueNodes.Get (i));
      sinkApp.Start (Seconds (0.5));
      sinkApp.Stop  (Seconds (simTime));

      // --- Client (OnOff) sur le RemoteHost vers ce UE ---
      OnOffHelper onOff ("ns3::UdpSocketFactory",
                         InetSocketAddress (ueIpAddrs.GetAddress (i), dlPort));
      onOff.SetAttribute ("DataRate",   DataRateValue (DataRate (dataRateStr)));
      onOff.SetAttribute ("PacketSize", UintegerValue (1400));
      onOff.SetAttribute ("OnTime",     StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
      onOff.SetAttribute ("OffTime",    StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

      ApplicationContainer clientApp = onOff.Install (remoteHost);
      // Démarrage légèrement décalé pour éviter la congestion initiale
      clientApp.Start (Seconds (1.0 + i * 0.02));
      clientApp.Stop  (Seconds (simTime - 1.0));
    }

  // ----------------------------------------------------------
  //  FLOW MONITOR — collecte des statistiques par flux
  // ----------------------------------------------------------
  FlowMonitorHelper flowMonHelper;
  Ptr<FlowMonitor> monitor = flowMonHelper.InstallAll ();
  Ptr<Ipv4FlowClassifier> classifier =
    DynamicCast<Ipv4FlowClassifier> (flowMonHelper.GetClassifier ());

  // ----------------------------------------------------------
  //  TRACES LTE (fichiers DlRlcStats.txt, UlRlcStats.txt, etc.)
  // ----------------------------------------------------------
  lteHelper->EnableTraces ();

  // ----------------------------------------------------------
  //  NETANIM — Animation XML
  // ----------------------------------------------------------

  // CORRECTION : Les nœuds internes EPC (PGW, SGW) et le RemoteHost
  // sont créés sans modèle de mobilité, ce qui génère des warnings.
  // On leur assigne une position fixe hors de la zone d'étude
  // pour qu'ils apparaissent dans NetAnim sans erreur.
  //
  // Position choisie : coin supérieur droit de la zone + offset
  double infraX = xMax + 500.0;
  double infraY = yMax + 500.0;

  // Parcourir TOUS les nœuds du simulateur pour trouver ceux sans mobilité
  // et leur assigner une position fixe. On identifie : PGW, SGW, RemoteHost.
  MobilityHelper fixedMobility;
  fixedMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  // Nœuds EPC internes (PGW = GetPgwNode, SGW interne, etc.)
  Ptr<Node> pgwNode = epcHelper->GetPgwNode ();
  if (!pgwNode->GetObject<MobilityModel> ())
    {
      fixedMobility.Install (pgwNode);
      pgwNode->GetObject<MobilityModel> ()->SetPosition (
        Vector (infraX, infraY, 0));
    }

  // RemoteHost
  if (!remoteHost->GetObject<MobilityModel> ())
    {
      fixedMobility.Install (remoteHost);
      remoteHost->GetObject<MobilityModel> ()->SetPosition (
        Vector (infraX + 300.0, infraY, 0));
    }

  // SGW et autres nœuds EPC internes : parcours défensif
  // (NS-3 EPC crée généralement 3 nœuds internes : SGW, PGW, MME)
  for (uint32_t ni = 0; ni < NodeList::GetNNodes (); ++ni)
    {
      Ptr<Node> n = NodeList::GetNode (ni);
      if (!n->GetObject<MobilityModel> ())
        {
          fixedMobility.Install (n);
          n->GetObject<MobilityModel> ()->SetPosition (
            Vector (infraX + 600.0 + ni * 200.0, infraY, 0));
        }
    }

  AnimationInterface anim ("anim_netmob_v04_50UE_5eNode.xml");

  // UE : rouge, taille proportionnelle à la zone
  double nodeSize = std::min (xMax - xMin, yMax - yMin) / 50.0;
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<MobilityModel> mob = ueNodes.Get (i)->GetObject<MobilityModel> ();
      Vector pos = mob ? mob->GetPosition () : Vector (0, 0, 0);
      anim.SetConstantPosition    (ueNodes.Get (i), pos.x, pos.y);
      anim.UpdateNodeDescription  (ueNodes.Get (i), "UE-" + std::to_string (i));
      anim.UpdateNodeColor        (ueNodes.Get (i), 220, 50, 50);   // Rouge
      anim.UpdateNodeSize         (ueNodes.Get (i), nodeSize, nodeSize);
    }

  // eNodeB : bleu, plus grands
  double enbSize = nodeSize * 2.0;
  for (uint32_t i = 0; i < nENodeB; ++i)
    {
      Vector pos = enbPositions[i];
      anim.SetConstantPosition    (enbNodes.Get (i), pos.x, pos.y);
      anim.UpdateNodeDescription  (enbNodes.Get (i), "eNB-" + std::to_string (i));
      anim.UpdateNodeColor        (enbNodes.Get (i), 30, 100, 200);  // Bleu
      anim.UpdateNodeSize         (enbNodes.Get (i), enbSize, enbSize);
    }

  // Nœuds infrastructure (PGW, RemoteHost) : gris, discrets
  anim.UpdateNodeDescription (pgwNode,    "PGW");
  anim.UpdateNodeColor       (pgwNode,    100, 100, 100);
  anim.UpdateNodeDescription (remoteHost, "RemoteHost");
  anim.UpdateNodeColor       (remoteHost, 80, 80, 80);

  anim.EnablePacketMetadata (true);

  // ----------------------------------------------------------
  //  PLANIFICATION DES MISES À JOUR PÉRIODIQUES
  //  La première mise à jour est à t=1s (trafic démarré)
  // ----------------------------------------------------------
  Simulator::Schedule (Seconds (interval), &PeriodicUpdate,
                       monitor, classifier, ueIpAddrs,
                       nNodes, ueNodes, interval, simTime);

  // ----------------------------------------------------------
  //  LANCEMENT DE LA SIMULATION
  // ----------------------------------------------------------
  std::cout << "\n==> Démarrage simulation..." << std::endl;
  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();

  // ----------------------------------------------------------
  //  RAPPORT FINAL — débit moyen global
  // ----------------------------------------------------------
  monitor->CheckForLostPackets ();
  monitor->SerializeToXmlFile ("flowmon_v04.xml", true, true);

  double totalThroughput = 0.0;
  uint32_t activeUe      = 0;
  for (auto& kv : g_ueStats)
    {
      if (kv.second.currentThroughputMbps > 0.0)
        {
          totalThroughput += kv.second.currentThroughputMbps;
          ++activeUe;
        }
    }

  std::cout << "\n======================================================" << std::endl;
  std::cout << "  RAPPORT FINAL" << std::endl;
  std::cout << "  UE actifs (avec trafic) : " << activeUe                << std::endl;
  std::cout << "  Débit total             : " << totalThroughput << " Mbps" << std::endl;
  if (activeUe > 0)
    std::cout << "  Débit moyen/UE          : "
              << totalThroughput / activeUe << " Mbps" << std::endl;
  std::cout << "======================================================" << std::endl;

  // ----------------------------------------------------------
  //  FERMETURE ET NETTOYAGE
  // ----------------------------------------------------------
  g_csvFile.close ();
  Simulator::Destroy ();

  std::cout << "\nSimulation terminée. Fichiers générés :" << std::endl;
  std::cout << "  - trajectories_v04_50UE_5eNode.csv" << std::endl;
  std::cout << "  - anim_netmob_v04_50UE_5eNode.xml" << std::endl;
  std::cout << "  - flowmon_v04.xml" << std::endl;
  std::cout << "  - DlRlcStats.txt, UlRlcStats.txt (traces LTE)" << std::endl;

  return 0;
}
