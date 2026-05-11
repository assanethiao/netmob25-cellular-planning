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

int main (int argc, char *argv[])
{
  uint32_t nNodes = 2;
  double simTime = 10.0;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nNodes", "Number of nodes", nNodes);
  cmd.AddValue ("simTime", "Simulation time (seconds)", simTime);
  cmd.Parse (argc, argv);

  std::cout << "=== LTE HIGH THROUGHPUT TEST ===" << std::endl;
//lteHelper->SetAttribute ("PathlossModel",                         StringValue ("ns3::FriisPropagationLossModel"));
  // 🔥 Puissance eNB
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (46.0));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // 🔥 Bande passante max
  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (100));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (100));

  // 🔥 Scheduler équitable
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (nNodes);

  // Mobilité Netmob
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::Netmob25MobilityModel",
                             "StartTime", TimeValue (Seconds (0.0)),
                             "UpdateInterval", TimeValue (Seconds (2)),
                             "ModelPath", StringValue ("model.pt"),
                             "TransportMode", StringValue ("WALKING"),
                             "TripLength", UintegerValue (100));
  mobility.Install (ueNodes);

for (uint32_t i = 0; i < ueNodes.GetN(); i++)
{
  Ptr<MobilityModel> mob = ueNodes.Get(i)->GetObject<MobilityModel>();
  mob->SetPosition(Vector(5*i, 0, 0)); // UEs proches
}
  // eNodeB fixe
  MobilityHelper mobilityEnb;
  mobilityEnb.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  Ptr<ListPositionAllocator> enbPos = CreateObject<ListPositionAllocator> ();
  enbPos->Add (Vector (0, 0, 0));
  mobilityEnb.SetPositionAllocator (enbPos);
  mobilityEnb.Install (enbNodes);

  // LTE devices
  NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueDevs = lteHelper->InstallUeDevice (ueNodes);

  InternetStackHelper internet;
  internet.Install (ueNodes);

  Ipv4InterfaceContainer ueIpAddrs;
  ueIpAddrs = epcHelper->AssignUeIpv4Address (ueDevs);

  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // =====================================================
  // 🔥 TRAFIC MULTI-UE (IMPORTANT)
  // =====================================================

  ApplicationContainer allApps;

  for (uint32_t i = 0; i < nNodes; ++i)
  {
    // Serveur sur chaque UE
    PacketSinkHelper sink ("ns3::TcpSocketFactory",
                           InetSocketAddress (Ipv4Address::GetAny (), 9000 + i));
    allApps.Add (sink.Install (ueNodes.Get (i)));

    // UE0 envoie vers tous les autres
    if (i != 0)
    {
      BulkSendHelper bulkSender ("ns3::TcpSocketFactory",
        InetSocketAddress (ueIpAddrs.GetAddress (i), 9000 + i));

      bulkSender.SetAttribute ("MaxBytes", UintegerValue (0));

      allApps.Add (bulkSender.Install (ueNodes.Get (0)));
    }
  }

  allApps.Start (Seconds (1.0));
  allApps.Stop (Seconds (simTime - 1));

  // Traces LTE
  lteHelper->EnableTraces ();

  std::cout << "Simulation en cours..." << std::endl;

  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  Simulator::Destroy ();

  std::cout << "Simulation terminée !" << std::endl;

  return 0;
}
