/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NetmobHighThroughput");

int main (int argc, char *argv[])
{
  uint32_t nUE = 50;
  uint32_t nEnb = 5;
  double simTime = 30.0;

  CommandLine cmd;
  cmd.AddValue ("nUE", "Number of UEs", nUE);
  cmd.AddValue ("nEnb", "Number of eNBs", nEnb);
  cmd.AddValue ("simTime", "Simulation time", simTime);
  cmd.Parse (argc, argv);

  std::cout << "=== LTE 50 UE / 5 eNB TEST ===" << std::endl;

  // 🔥 CONFIG RADIO
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (46.0));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (100));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (100));

  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  lteHelper->SetAttribute ("PathlossModel",
                           StringValue ("ns3::FriisPropagationLossModel"));

  // 🔥 NODES
  NodeContainer ueNodes;
  NodeContainer enbNodes;

  ueNodes.Create (nUE);
  enbNodes.Create (nEnb);

  // ===============================
  // 📡 POSITION eNodeB (grille)
  // ===============================
  MobilityHelper mobilityEnb;
  mobilityEnb.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  Ptr<ListPositionAllocator> enbAlloc = CreateObject<ListPositionAllocator> ();

  enbAlloc->Add (Vector (0, 0, 0));
  enbAlloc->Add (Vector (200, 0, 0));
  enbAlloc->Add (Vector (0, 200, 0));
  enbAlloc->Add (Vector (200, 200, 0));
  enbAlloc->Add (Vector (100, 100, 0));

  mobilityEnb.SetPositionAllocator (enbAlloc);
  mobilityEnb.Install (enbNodes);

  // ===============================
  // 📱 POSITION UE (répartition)
  // ===============================
  MobilityHelper mobilityUe;
  mobilityUe.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  Ptr<ListPositionAllocator> ueAlloc = CreateObject<ListPositionAllocator> ();

  for (uint32_t i = 0; i < nUE; i++)
  {
    double x = (i % 10) * 20 + 10;
    double y = (i / 10) * 20 + 10;
    ueAlloc->Add (Vector (x, y, 0));
  }

  mobilityUe.SetPositionAllocator (ueAlloc);
  mobilityUe.Install (ueNodes);

  // ===============================
  // 📡 INSTALL DEVICES
  // ===============================
  NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueDevs = lteHelper->InstallUeDevice (ueNodes);

  // ===============================
  // 🌐 INTERNET
  // ===============================
  InternetStackHelper internet;
  internet.Install (ueNodes);

  Ipv4InterfaceContainer ueIp = epcHelper->AssignUeIpv4Address (ueDevs);

  // 🔥 ATTACH AUTO (important multi-eNB)
  for (uint32_t i = 0; i < nUE; i++)
  {
    lteHelper->AttachToClosestEnb (ueDevs.Get(i), enbDevs);
  }

  // ===============================
  // 🔥 TRAFIC TCP SATURÉ
  // ===============================
  uint16_t port = 9;

  // Serveur = UE 0
  PacketSinkHelper sink ("ns3::TcpSocketFactory",
                         InetSocketAddress (Ipv4Address::GetAny (), port));

  ApplicationContainer sinkApp = sink.Install (ueNodes.Get (0));
  sinkApp.Start (Seconds (1.0));
  sinkApp.Stop (Seconds (simTime));

  // Tous les autres UE envoient vers UE0
  for (uint32_t i = 1; i < nUE; i++)
  {
    BulkSendHelper sender ("ns3::TcpSocketFactory",
                           InetSocketAddress (ueIp.GetAddress (0), port));

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
