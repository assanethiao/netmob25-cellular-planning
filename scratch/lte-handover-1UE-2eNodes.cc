#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/netanim-module.h"
#include "ns3/applications-module.h"
#include <fstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LteHandoverForced");

std::ofstream csvFile;

// ===================== LOG DATA =====================
void LogData(Ptr<Node> ueNode, NetDeviceContainer enbDevs) {
    Ptr<MobilityModel> ueMob = ueNode->GetObject<MobilityModel>();
    Vector uePos = ueMob->GetPosition();

    // Calcul du débit basé sur l'antenne la plus proche (Simulé)
    double maxThroughput = 0.0;
    int bestEnb = 0;
    double d0 = 150.0; // Portée de référence

    for (uint32_t i = 0; i < enbDevs.GetN(); i++) {
        Vector enbPos = enbDevs.Get(i)->GetNode()->GetObject<MobilityModel>()->GetPosition();
        double dist = CalculateDistance(uePos, enbPos);
        
        // Formule de débit variable : 50 Mbps max au pied de l'antenne
        double currentThroughput = 50.0 * std::exp(-dist / d0);
        if (currentThroughput > maxThroughput) {
            maxThroughput = currentThroughput;
            bestEnb = i;
        }
    }

    double time = Simulator::Now().GetSeconds();
    csvFile << time << "," << uePos.x << "," << uePos.y << "," << bestEnb << "," << maxThroughput << std::endl;

    Simulator::Schedule(Seconds(0.5), &LogData, ueNode, enbDevs);
}

int main(int argc, char *argv[]) {
    // Réglage du temps de simulation
    Time simTime = Seconds(60);

    NodeContainer enbNodes;
    enbNodes.Create(2);
    NodeContainer ueNodes;
    ueNodes.Create(1);

    // ===================== MOBILITÉ eNB =====================
    MobilityHelper mobilityEnb;
    mobilityEnb.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobilityEnb.Install(enbNodes);
    enbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0, 0, 0));
    enbNodes.Get(1)->GetObject<MobilityModel>()->SetPosition(Vector(300, 0, 0));

    // ===================== MOBILITÉ UE (Trajet Forcé) =====================
    // On utilise Waypoints pour forcer l'UE à aller de eNB 1 vers eNB 2
    Ptr<WaypointMobilityModel> waypoints = CreateObject<WaypointMobilityModel>();
    waypoints->AddWaypoint(Waypoint(Seconds(0.0), Vector(-50, 20, 0)));   // Départ proche eNB 1
    waypoints->AddWaypoint(Waypoint(Seconds(30.0), Vector(350, 20, 0)));  // Arrivée proche eNB 2
    waypoints->AddWaypoint(Waypoint(Seconds(60.0), Vector(-50, 20, 0)));  // Retour
    ueNodes.Get(0)->AggregateObject(waypoints);

    // ===================== LTE + EPC =====================
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    // Paramètres A3 agressifs pour forcer le handover rapidement
    lteHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    lteHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(1.0)); // Faible hystérésis = handover facile
    lteHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(128)));

    NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs = lteHelper->InstallUeDevice(ueNodes);

    InternetStackHelper internet;
    internet.Install(ueNodes);
    epcHelper->AssignUeIpv4Address(ueDevs);

    // Attachement initial
    lteHelper->Attach(ueDevs.Get(0), enbDevs.Get(0));

    // ===================== SORTIES =====================
    csvFile.open("lte-handover-1UE-2eNodes-v01.csv");
    csvFile << "Time,X,Y,Best_eNB,Throughput_Mbps" << std::endl;
    Simulator::Schedule(Seconds(0.1), &LogData, ueNodes.Get(0), enbDevs);

    AnimationInterface anim("lte-handover-1ue-2eNodes-v01.xml");
    anim.UpdateNodeDescription(0, "eNB-1 (Cell-1)");
    anim.UpdateNodeDescription(1, "eNB-2 (Cell-2)");

    Simulator::Stop(simTime);
    Simulator::Run();
    
    csvFile.close();
    Simulator::Destroy();
    return 0;
}
