/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/netmob25-mobility-model.h"
#include "ns3/applications-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NetmobHighThroughput");

int
main (int argc, char *argv[])
{
  uint32_t nNodes = 3;
  double simTime = 30.0;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nNodes", "Number of nodes", nNodes);
  cmd.AddValue ("simTime", "Simulation time", simTime);
  cmd.Parse (argc, argv);

  std::cout << "=== LTE HIGH THROUGHPUT TEST ===" << std::endl;

  // 🔥 Puissance émission
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (46.0));

  // 🔥 Création helper AVANT utilisation
  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // 🔥 Bande max LTE
  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (100));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (100));

  // 🔥 Bon canal radio
  lteHelper->SetAttribute ("PathlossModel",
                           StringValue ("ns3::FriisPropagationLossModel"));

  // 🔥 Scheduler performant
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  // Nodes
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (nNodes);

  // Mobilité simple (meilleur débit)
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (ueNodes);

  Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator> ();
  posAlloc->Add (Vector (5, 0, 0));
  posAlloc->Add (Vector (10, 0, 0));
  posAlloc->Add (Vector (15, 0, 0));
  mobility.SetPositionAllocator (posAlloc);

  // eNodeB
  MobilityHelper mobilityEnb;
  mobilityEnb.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  Ptr<ListPositionAllocator> enbAlloc = CreateObject<ListPositionAllocator> ();
  enbAlloc->Add (Vector (0, 0, 0));

  mobilityEnb.SetPositionAllocator (enbAlloc);
  mobilityEnb.Install (enbNodes);

  // Devices
  NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueDevs = lteHelper->InstallUeDevice (ueNodes);

  // Internet
  InternetStackHelper internet;
  internet.Install (ueNodes);

  Ipv4InterfaceContainer ueIp = epcHelper->AssignUeIpv4Address (ueDevs);

  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // ===============================
  // 🔥 TRAFIC TCP SATURÉ
  // ===============================

  uint16_t port = 9;

  // Serveur (UE 1)
  PacketSinkHelper sink ("ns3::TcpSocketFactory",
                         InetSocketAddress (Ipv4Address::GetAny (), port));

  ApplicationContainer sinkApp = sink.Install (ueNodes.Get (1));
  sinkApp.Start (Seconds (1.0));
  sinkApp.Stop (Seconds (simTime));

  // Clients (tous les autres UE → UE1)
  for (uint32_t i = 0; i < nNodes; i++)
  {
    if (i == 1) continue;

    BulkSendHelper sender ("ns3::TcpSocketFactory",
                           InetSocketAddress (ueIp.GetAddress (1), port));

    sender.SetAttribute ("MaxBytes", UintegerValue (0));

    ApplicationContainer app = sender.Install (ueNodes.Get (i));
    app.Start (Seconds (1.0));
    app.Stop (Seconds (simTime - 1));
  }

  // Traces
  lteHelper->EnableTraces ();

  std::cout << "Simulation en cours..." << std::endl;

  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  Simulator::Destroy ();

  std::cout << "Simulation terminée !" << std::endl;

  return 0;
}
