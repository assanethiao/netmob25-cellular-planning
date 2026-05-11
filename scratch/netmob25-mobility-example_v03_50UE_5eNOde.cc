/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Simple test for Netmob25 trajectory generation. In this simultion, an eNodeB is placed in the middle of the mobility area of the mobile node. 
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/netmob25-mobility-model.h"
#include "ns3/applications-module.h"
#include <fstream>
#include "ns3/netanim-module.h"


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Netmob25SimpleTestv2");

int
main (int argc, char *argv[])
{
  uint32_t nNodes = 50;
  uint32_t nENodeB = 5; 
  double simTime = 100.0;
  double x_min = 10000.0, x_max = -10000.0, y_min = 10000.0, y_max = -10000.0;
  
  CommandLine cmd (__FILE__);
  cmd.AddValue ("nNodes", "Number of nodes", nNodes);
  cmd.AddValue ("nENodeB", "Number of eNodeB", nENodeB);
  cmd.AddValue ("simTime", "Simulation time (seconds)", simTime);
  cmd.Parse (argc, argv);
  

  // Enable debug logging
  LogComponentEnable ("Netmob25MobilityModel", LOG_LEVEL_INFO);

  std::cout << "=== Netmob25 Simple Trajectory Test ===" << std::endl;
  std::cout << "Testing trajectory generation with " << nNodes << " nodes" << std::endl;
  std::cout << "Simulation time: " << simTime << " seconds" << std::endl;
  std::cout << "Model path: model.pt" << std::endl;
  std::cout << "========================================" << std::endl;

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);
  //lteHelper->SetAttribute("PathlossModel", StringValue("ns3::Cost231PropagationLossModel"));
  
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (nENodeB);
  ueNodes.Create (nNodes);
 
  std::ofstream file("trajectories_v03_50UE_5eNode.csv");
  file << "time,node,x,y,z\n";


  // Setup mobility with Netmob25MobilityModel
  MobilityHelper mobility;
  
  // Configure the model with explicit parameters
  mobility.SetMobilityModel ("ns3::Netmob25MobilityModel",
                            "StartTime", TimeValue (Seconds (0.0)),
                            "UpdateInterval", TimeValue (Seconds (2)),  // Update every 2s
                            "ModelPath", StringValue ("model.pt"),
                            "TransportMode", StringValue ("WALKING"),
                            "TripLength", UintegerValue (100));
  
  mobility.Install (ueNodes);

  // Position printing function
  /*auto printPositions = [&ueNodes, nNodes, &x_min, &x_max, &y_min, &y_max]() {
    std::cout << "\n[" << Simulator::Now ().GetSeconds () << "s] Node positions:" << std::endl;
    for (uint32_t i = 0; i < nNodes; ++i)
      {
        Ptr<MobilityModel> mob = ueNodes.Get (i)->GetObject<MobilityModel> ();
        if (mob)
          {
            Vector pos = mob->GetPosition ();
            Vector vel = mob->GetVelocity ();
            std::cout << "  Node " << i << ": Position(" << pos.x << ", " << pos.y << ", " << pos.z 
                      << ") Velocity(" << vel.x << ", " << vel.y << ", " << vel.z << ")" << std::endl;
            if (pos.x < x_min) {*(&x_min) = (double)pos.x;}
            if (pos.x > x_max) {*(&x_max) = (double)pos.x;}
            if (pos.y < y_min) {*(&y_min) = pos.y;}
            if (pos.y > y_max) {*(&y_max) = pos.y;}
          }
        else
          {
            std::cout << "  Node " << i << ": No mobility model!" << std::endl;
          }
      }
  };  */


  auto printPositions = [&ueNodes, nNodes, &x_min, &x_max, &y_min, &y_max, &file]() {
    double t = Simulator::Now ().GetSeconds ();
    std::cout << "\n[" << t << "s] Node positions:" << std::endl;

    for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<MobilityModel> mob = ueNodes.Get (i)->GetObject<MobilityModel> ();
      if (mob)
      {
        Vector pos = mob->GetPosition ();
        Vector vel = mob->GetVelocity ();

         std::cout << "  Node " << i << ": Position(" << pos.x << ", " << pos.y << ", " << pos.z 
                  << ") Velocity(" << vel.x << ", " << vel.y << ", " << vel.z << ")" << std::endl;

        // 👉 Écriture dans le CSV
        file << t << "," << i << "," << pos.x << "," << pos.y << "," << pos.z << "\n";

        // Mise à jour min/max
        if (pos.x < x_min) x_min = pos.x;
        if (pos.x > x_max) x_max = pos.x;
        if (pos.y < y_min) y_min = pos.y;
        if (pos.y > y_max) y_max = pos.y;
      }
    }
  };




  // Print initial positions
  std::cout << "\nInitial positions:" << std::endl;
  printPositions();

  // Schedule position printing every second
  for (double t = 1.0; t <= simTime; t += 1.0)
    {
      Simulator::Schedule (Seconds (t), printPositions);
    }
    
  
  
  MobilityHelper mobility1;
  mobility1.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility1.Install (enbNodes);
  //std::cout << "x_min = " << x_min << ", x_max = " << x_max << ", y_min = " << y_min << ", y_max = " << y_max << "\n";
  
  Simulator::Schedule(Seconds(5.0), [&]() {
    std::cout << "UPDATED bounds: "
            << x_min << " " << x_max << " "
            << y_min << " " << y_max << std::endl;
  });  

  Vector enb_pos((x_max + x_min)/2, (y_max + y_min)/2, 0);
/*  std::cout << "  eNodeB : Position(" << enb_pos.x << ", " << enb_pos.y << ", " << enb_pos.z << ")" << std::endl;
  Ptr<ListPositionAllocator> positionAllocator = CreateObject<ListPositionAllocator>();
  positionAllocator->Add(enb_pos);
  mobility1.SetPositionAllocator(positionAllocator);
  mobility1.Install (enbNodes);
  Ptr<MobilityModel> mobilityModel = enbNodes.Get(0)->GetObject<MobilityModel>();
  enb_pos = mobilityModel->GetPosition();
  std::cout << "  eNodeB : verified Position(" << enb_pos.x << ", " << enb_pos.y << ", " << enb_pos.z << ")" << std::endl;
  */

  Ptr<ListPositionAllocator> enbPositionAllocator = CreateObject<ListPositionAllocator>();
  enbPositionAllocator->Add(Vector(x_min + 100, y_min + 100, 0));
  enbPositionAllocator->Add(Vector(x_max - 100, y_min + 100, 0));
  enbPositionAllocator->Add(Vector(x_min + 100, y_max - 100, 0));
  enbPositionAllocator->Add(Vector(x_max - 100, y_max - 100, 0));
  enbPositionAllocator->Add(Vector((x_max + x_min)/2, (y_max + y_min)/2, 0));

  MobilityHelper enbMobility;
  enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  enbMobility.SetPositionAllocator(enbPositionAllocator);
  enbMobility.Install(enbNodes);

  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);
  
  InternetStackHelper tcpip;
  tcpip.Install (ueNodes);
  
  Ipv4InterfaceContainer ueIpAddrs;
  ueIpAddrs = epcHelper->AssignUeIpv4Address (ueDevs);
  
  //lteHelper->Attach (ueDevs, enbDevs.Get (0));

  for (uint32_t i = 0; i < ueDevs.GetN(); ++i)
  {
      uint32_t enbIndex = i % nENodeB;  // distribuer sur 5 eNodeB
      lteHelper->Attach(ueDevs.Get(i), enbDevs.Get(enbIndex));
  }  


 /* UdpEchoServerHelper echoServer(9);
  ApplicationContainer serverApps = echoServer.Install (ueNodes.Get(1));
  serverApps.Start (Seconds(1.0));
  serverApps.Stop (Seconds(41.0));
  UdpEchoClientHelper echoClient (ueIpAddrs.GetAddress(1), 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue(10));
  echoClient.SetAttribute ("Interval", TimeValue(Seconds (1.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue(1024));
  ApplicationContainer clientApps = echoClient.Install (ueNodes.Get(0));
  clientApps.Start (Seconds (1.0));
  clientApps.Stop (Seconds (40.0));
*/

// Installer un serveur sur chaque eNodeB
for (uint32_t i = 0; i < nENodeB; ++i)
{
    UdpEchoServerHelper echoServer(9 + i);  // ports différents
    ApplicationContainer serverApps = echoServer.Install(ueNodes.Get(i)); // ou un subset de UE
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(simTime - 10));
}

// Installer clients sur tous les UE
  for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
  {
      uint32_t serverPort = 9 + (i % nENodeB);
      UdpEchoClientHelper echoClient(ueIpAddrs.GetAddress(i % ueNodes.GetN()), serverPort);
      echoClient.SetAttribute("MaxPackets", UintegerValue(100));
      echoClient.SetAttribute("Interval", TimeValue(Seconds(1.0)));
      echoClient.SetAttribute("PacketSize", UintegerValue(1024));
      ApplicationContainer clientApps = echoClient.Install(ueNodes.Get(i));
      clientApps.Start(Seconds(1.0));
      clientApps.Stop(Seconds(simTime - 10));
  }  


  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
  lteHelper->EnableTraces ();
  


  //netanim
  AnimationInterface anim("anim_netmob_v03_50UE_5eNode.xml");


  // IMPORTANT : définir les positions initiales
  for (uint32_t i = 0; i < nNodes; ++i)
  {
    Ptr<MobilityModel> mob = ueNodes.Get(i)->GetObject<MobilityModel>();
    Vector pos = mob->GetPosition();

    anim.SetConstantPosition(ueNodes.Get(i), pos.x, pos.y);

    anim.UpdateNodeDescription(ueNodes.Get(i), "UE-" + std::to_string(i));
    anim.UpdateNodeColor(ueNodes.Get(i), 255, 0, 0); // 🔴 rouge
    anim.UpdateNodeSize(ueNodes.Get(i), 220.0, 220.0); // taille UE
  }

  // eNodeB
  /*Ptr<MobilityModel> enbMob = enbNodes.Get(0)->GetObject<MobilityModel>();
  Vector enbPos = enbMob->GetPosition();

  anim.SetConstantPosition(enbNodes.Get(0), enbPos.x, enbPos.y);

  anim.UpdateNodeDescription(enbNodes.Get(0), "eNodeB");
  anim.UpdateNodeColor(enbNodes.Get(0), 0, 0, 255); // 🔵 bleu
  anim.UpdateNodeSize(enbNodes.Get(0), 160.0, 160.0); // plus grand

*/

  for (uint32_t i = 0; i < nENodeB; ++i)
  {
      Ptr<MobilityModel> enbMob = enbNodes.Get(i)->GetObject<MobilityModel>();
      Vector enbPos = enbMob->GetPosition();
      anim.SetConstantPosition(enbNodes.Get(i), enbPos.x, enbPos.y);
      anim.UpdateNodeDescription(enbNodes.Get(i), "eNodeB-" + std::to_string(i));
      anim.UpdateNodeColor(enbNodes.Get(i), 0, 0, 255); // bleu
      anim.UpdateNodeSize(enbNodes.Get(i), 250.0, 250.0);
  }

  // Optionnel mais utile
  anim.EnablePacketMetadata(true);

  for (uint32_t i = 0; i < nNodes; ++i)
  {
    anim.UpdateNodeDescription(ueNodes.Get(i), "UE-" + std::to_string(i));
  }

  anim.UpdateNodeDescription(enbNodes.Get(0), "eNodeB");
  // Run simulation
  std::cout << "\nStarting simulation..." << std::endl;
  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  
  // Print final positions
  std::cout << "\nFinal positions:" << std::endl;
  printPositions();

  file.close(); 

  Simulator::Destroy ();

  std::cout << "\nSimulation completed!" << std::endl;
  
  return 0;
}
