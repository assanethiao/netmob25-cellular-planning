#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/node-list.h"
#include <fstream>
#include <map>
#include <cmath>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("LteHandoverFixed");

// ================= GLOBAL =================
std::ofstream csvFile;
std::map<uint16_t, double> g_rsrp;
uint16_t g_connectedCell = 1;
uint64_t g_rxBytes = 0;
double g_lastLogTime = 0.0;

// ================= CALLBACKS =================
void RsrpSinrCallback(std::string context,
                      uint16_t cellId, uint16_t rnti,
                      double rsrp, double sinr, uint8_t ccId)
{
    g_rsrp[cellId] = rsrp;
}

void HandoverStart(std::string context,
                   uint64_t imsi,
                   uint16_t cellId,
                   uint16_t rnti,
                   uint16_t targetCellId)
{
    std::cout << Simulator::Now().GetSeconds()
              << "s  HO START: " << cellId
              << " → " << targetCellId << std::endl;
}

void HandoverEnd(std::string context,
                 uint64_t imsi, uint16_t cellId,
                 uint16_t rnti)
{
    g_connectedCell = cellId;
    std::cout << Simulator::Now().GetSeconds()
              << "s  *** HANDOVER COMPLETE → Cell "
              << cellId << " ***" << std::endl;
}

void RxCallback(Ptr<const Packet> pkt, const Address &addr)
{
    g_rxBytes += pkt->GetSize();
}

// ================= UTILS =================
double RsrpToThroughput(double rsrp_W)
{
    if (rsrp_W <= 0.0) return 0.0;

    double rsrp_dBm = 10.0 * std::log10(rsrp_W * 1000.0);
    double sinr_dB  = rsrp_dBm - (-100.0);
    double sinr_lin = std::pow(10.0, sinr_dB / 10.0);

    double cap = 20e6 * std::log2(1 + sinr_lin) / 1e6;
    return std::min(cap, 150.0);
}

// ================= LOG =================
void LogData(Ptr<Node> ueNode)
{
    double now = Simulator::Now().GetSeconds();
    Vector pos = ueNode->GetObject<MobilityModel>()->GetPosition();

    double dt = now - g_lastLogTime;
    double thr = (dt > 0) ? (g_rxBytes * 8.0 / dt / 1e6) : 0.0;

    g_rxBytes = 0;
    g_lastLogTime = now;

    double r1 = g_rsrp.count(1) ? g_rsrp[1] : 0;
    double r2 = g_rsrp.count(2) ? g_rsrp[2] : 0;

    auto toDbm = [](double w) {
        return (w > 0) ? 10 * log10(w * 1000) : -999.0;
    };

    csvFile << now << ","
            << pos.x << "," << pos.y << ","
            << toDbm(r1) << "," << RsrpToThroughput(r1) << ","
            << toDbm(r2) << "," << RsrpToThroughput(r2) << ","
            << g_connectedCell << ","
            << thr << "\n";

    Simulator::Schedule(Seconds(0.5), &LogData, ueNode);
}

// ================= MAIN =================
int main(int argc, char *argv[])
{
    double simTime = 120.0;
    CommandLine cmd;
    cmd.Parse(argc, argv);

    // RANDOM
    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(1);

    // LTE
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    // ⭐ CRITIQUE : active handover correctement
    lteHelper->SetAttribute("UseIdealRrc", BooleanValue(true));

    // Propagation réaliste
    lteHelper->SetAttribute("PathlossModel",
        StringValue("ns3::LogDistancePropagationLossModel"));

    Config::SetDefault("ns3::LogDistancePropagationLossModel::Exponent",
                       DoubleValue(3.5));

    // Handover A3
    lteHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    lteHelper->SetHandoverAlgorithmAttribute("Hysteresis",
                                             DoubleValue(0.5));
    lteHelper->SetHandoverAlgorithmAttribute("TimeToTrigger",
                                             TimeValue(MilliSeconds(64)));

    // NODES
    NodeContainer enbNodes; enbNodes.Create(2);
    NodeContainer ueNodes; ueNodes.Create(1);

    // eNB positions
    MobilityHelper mobEnb;
    mobEnb.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobEnb.Install(enbNodes);

    enbNodes.Get(0)->GetObject<MobilityModel>()
        ->SetPosition(Vector(0, 150, 25));

    enbNodes.Get(1)->GetObject<MobilityModel>()
        ->SetPosition(Vector(300, 150, 25));

    // UE mobility
    Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
    posAlloc->Add(Vector(150, 150, 0));

    MobilityHelper mobUe;
    mobUe.SetPositionAllocator(posAlloc);
    mobUe.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
        "Bounds", RectangleValue(Rectangle(0, 300, 0, 300)),
        "Speed", StringValue("ns3::UniformRandomVariable[Min=10|Max=20]"),
        "Distance", DoubleValue(50.0),
        "Mode", StringValue("Distance"));
    mobUe.Install(ueNodes);

    // DEVICES
    NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs  = lteHelper->InstallUeDevice(ueNodes);

    // INTERNET
    InternetStackHelper internet;
    internet.Install(ueNodes);

    Ipv4InterfaceContainer ueIp =
        epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    Ipv4StaticRoutingHelper routing;
    routing.GetStaticRouting(ueNodes.Get(0)->GetObject<Ipv4>())
        ->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

    // REMOTE HOST
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    internet.Install(remoteHostContainer);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("5ms"));

    NetDeviceContainer internetDevices =
        p2p.Install(epcHelper->GetPgwNode(), remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    ipv4h.Assign(internetDevices);

    // APP
    uint16_t port = 5000;

    PacketSinkHelper sink("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));

    ApplicationContainer sinkApp = sink.Install(ueNodes.Get(0));
    sinkApp.Start(Seconds(1.0));
    sinkApp.Stop(Seconds(simTime));

    sinkApp.Get(0)->TraceConnectWithoutContext("Rx",
        MakeCallback(&RxCallback));

    UdpClientHelper client(ueIp.GetAddress(0), port);
    client.SetAttribute("Interval", TimeValue(MilliSeconds(1)));
    client.SetAttribute("PacketSize", UintegerValue(1400));
    client.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));

    ApplicationContainer clientApp = client.Install(remoteHost);
    clientApp.Start(Seconds(1.5));
    clientApp.Stop(Seconds(simTime));

    // ATTACH
    lteHelper->Attach(ueDevs.Get(0), enbDevs.Get(0));

    // CALLBACKS
    Config::Connect(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/LteUePhy/ReportCurrentCellRsrpSinr",
        MakeCallback(&RsrpSinrCallback));

    Config::Connect(
        "/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
        MakeCallback(&HandoverStart));

    Config::Connect(
        "/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
        MakeCallback(&HandoverEnd));

    // CSV
    csvFile.open("lte-final.csv");
    csvFile << "Time,X,Y,RSRP1,Thr1,RSRP2,Thr2,Cell,ActualThr\n";

    Simulator::Schedule(Seconds(1.0), &LogData, ueNodes.Get(0));

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    csvFile.close();
    Simulator::Destroy();
}
