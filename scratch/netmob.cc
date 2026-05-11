/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/netmob25-mobility-model.h"
#include "ns3/applications-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Netmob25SimpleTestv2");

int
main (int argc, char *argv[])
{
  uint32_t nNodes = 2;
  double simTime = 10.0;
  double x_min = 100000.0, x_max = -100000.0, y_min = 100000.0, y_max = -100000.0;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nNodes", "Number of nodes", nNodes);
  cmd.AddValue ("simTime", "Simulation time (seconds)", simTime);
  cmd.Parse (argc, argv);

  LogComponentEnable ("Netmob25MobilityModel", LOG_LEVEL_INFO);

  std::cout << "=== LTE HIGH THROUGHPUT TEST ===" << std::endl;

  // 🔥 CONFIG LTE (IMPORTANT)
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (46.0));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // 🔥 Bande passante max (20 MHz)
  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (100));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (100));

  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (nNodes);

  // Mobilité Netmob (inchangée)
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::Netmob25MobilityModel",
                            "StartTime", TimeValue (Seconds (0.0)),
                            "UpdateInterval", TimeValue (Seconds (2)),
                            "ModelPath", StringValue ("model.pt"),
                            "TransportMode", StringValue ("WALKING"),
                            "TripLength", UintegerValue (100));
  mobility.Install (ueNodes);

  // eNodeB position
  MobilityHelper mobility1;
  mobility1.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility1.Install (enbNodes);

  Ptr<ListPositionAllocator> positionAllocator = CreateObject<ListPositionAllocator>();
  positionAllocator->Add(Vector(0,0,0));
  mobility1.SetPositionAllocator(positionAllocator);
  mobility1.Install (enbNodes);

  // Devices LTE
  NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueDevs = lteHelper->InstallUeDevice (ueNodes);

  InternetStackHelper tcpip;
  tcpip.Install (ueNodes);

  Ipv4InterfaceContainer ueIpAddrs;
  ueIpAddrs = epcHelper->AssignUeIpv4Address (ueDevs);

  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // =====================================================
  // 🔥 TRAFIC TCP SATURÉ (REMPLACE UDP ECHO)
  // =====================================================

  // Serveur (UE 1)
  PacketSinkHelper sink ("ns3::TcpSocketFactory",
                         InetSocketAddress (Ipv4Address::GetAny (), 9));

  ApplicationContainer sinkApp = sink.Install (ueNodes.Get(1));
  sinkApp.Start (Seconds (1.0));
  sinkApp.Stop (Seconds (simTime));

  // Client saturé (UE 0)
  BulkSendHelper bulkSender ("ns3::TcpSocketFactory",
                             InetSocketAddress (ueIpAddrs.GetAddress(1), 9));

  bulkSender.SetAttribute ("MaxBytes", UintegerValue (0)); // infini

  ApplicationContainer senderApp = bulkSender.Install (ueNodes.Get(0));
  senderApp.Start (Seconds (1.0));
  senderApp.Stop (Seconds (simTime - 1));

  // Traces
  lteHelper->EnableTraces ();

  std::cout << "Simulation en cours..." << std::endl;

  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  Simulator::Destroy ();

  std::cout << "Simulation terminée !" << std::endl;

  return 0;
}
