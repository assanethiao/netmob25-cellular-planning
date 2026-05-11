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
NS_LOG_COMPONENT_DEFINE("LteHandoverRandom");

// ── État global ─────────────────────────────────────────
std::ofstream csvFile;
std::map<uint16_t, double> g_rsrp;
uint16_t g_connectedCell = 1;
uint64_t g_rxBytes = 0;
double g_lastLogTime = 0.0;

// ── Callback RSRP ───────────────────────────────────────
void RsrpSinrCallback(std::string context,
                      uint16_t cellId, uint16_t rnti,
                      double rsrp, double sinr,
                      uint8_t componentCarrierId)
{
    g_rsrp[cellId] = rsrp;
}

// ── Callback Handover ───────────────────────────────────
void HandoverEnd(std::string context,
                 uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    g_connectedCell = cellId;
    NS_LOG_UNCOND(Simulator::Now().GetSeconds()
                  << "s HANDOVER → Cell " << cellId);
}

// ── RX callback ─────────────────────────────────────────
void RxCallback(Ptr<const Packet> pkt, const Address &addr)
{
    g_rxBytes += pkt->GetSize();
}

// ── Modèle débit ────────────────────────────────────────
double RsrpToThroughput(double rsrp_W)
{
    if (rsrp_W <= 0) return 0.0;

    double rsrp_dBm = 10.0 * std::log10(rsrp_W * 1000.0);
    double noise_dBm = -100.0;

    double sinr_dB = rsrp_dBm - noise_dBm;
    double sinr_lin = std::pow(10.0, sinr_dB / 10.0);

    double bw = 20e6;
    double capacity = bw * std::log2(1.0 + sinr_lin) / 1e6;

    return std::min(capacity, 150.0);
}

// ── Logging ─────────────────────────────────────────────
void LogData(Ptr<Node> ueNode)
{
    double now = Simulator::Now().GetSeconds();
    Vector pos = ueNode->GetObject<MobilityModel>()->GetPosition();

    double dt = now - g_lastLogTime;
    double actualThroughput = (dt > 0)
        ? (g_rxBytes * 8.0 / dt / 1e6)
        : 0.0;

    g_rxBytes = 0;
    g_lastLogTime = now;

    double rsrp1 = g_rsrp.count(1) ? g_rsrp[1] : 0.0;
    double rsrp2 = g_rsrp.count(2) ? g_rsrp[2] : 0.0;

    double thr1 = RsrpToThroughput(rsrp1);
    double thr2 = RsrpToThroughput(rsrp2);

    // DEBUG console
    std::cout << "t=" << now
              << " RSRP1=" << (rsrp1 > 0 ? 10.0*log10(rsrp1*1000.0) : -999)
              << " RSRP2=" << (rsrp2 > 0 ? 10.0*log10(rsrp2*1000.0) : -999)
              << std::endl;

    csvFile << now << ","
            << pos.x << ","
            << pos.y << ","
            << (rsrp1 > 0 ? 10.0*log10(rsrp1*1000.0) : -999) << ","
            << thr1 << ","
            << (rsrp2 > 0 ? 10.0*log10(rsrp2*1000.0) : -999) << ","
            << thr2 << ","
            << g_connectedCell << ","
            << actualThroughput << "\n";

    Simulator::Schedule(Seconds(0.5), &LogData, ueNode);
}

// ── MAIN ────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    double simTime = 120.0;
    CommandLine cmd;
    cmd.AddValue("simTime", "Durée (s)", simTime);
    cmd.Parse(argc, argv);

    // LTE
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    lteHelper->SetAttribute("PathlossModel",
        StringValue("ns3::LogDistancePropagationLossModel"));

    lteHelper->SetSchedulerType("ns3::PfFfMacScheduler");

    // Handover amélioré
    lteHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    lteHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(1.0));
    lteHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(100)));

    // Nodes
    NodeContainer enbNodes; enbNodes.Create(2);
    NodeContainer ueNodes; ueNodes.Create(1);

    // eNB mobilité
    MobilityHelper mobEnb;
    mobEnb.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobEnb.Install(enbNodes);

    enbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(100, 250, 30));
    enbNodes.Get(1)->GetObject<MobilityModel>()->SetPosition(Vector(300, 250, 30));

    // UE mobilité (améliorée)
    MobilityHelper mobUe;
    mobUe.SetMobilityModel("ns3::RandomDirection2dMobilityModel",
        "Bounds", RectangleValue(Rectangle(0, 500, 0, 500)),
        "Speed", StringValue("ns3::UniformRandomVariable[Min=5.0|Max=20.0]"),
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.5]")
    );
    mobUe.Install(ueNodes);

    ueNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(200, 250, 1.5));

    // Devices
    NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs = lteHelper->InstallUeDevice(ueNodes);

    // Internet
    InternetStackHelper internet;
    internet.Install(ueNodes);

    Ipv4InterfaceContainer ueIpIface =
        epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> ueRoute =
        ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(0)->GetObject<Ipv4>());
    ueRoute->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

    // Remote host
    NodeContainer remoteHostContainer; remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    internet.Install(remoteHostContainer);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));

    NetDeviceContainer internetDevices =
        p2p.Install(epcHelper->GetPgwNode(), remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIfaces =
        ipv4h.Assign(internetDevices);

    Ptr<Ipv4StaticRouting> remoteRoute =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteRoute->AddNetworkRouteTo(
        Ipv4Address("7.0.0.0"),
        Ipv4Mask("255.0.0.0"), 1);

    // Application
    uint16_t port = 5000;

    PacketSinkHelper sink("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));

    ApplicationContainer sinkApps = sink.Install(ueNodes.Get(0));
    sinkApps.Start(Seconds(1.0));
    sinkApps.Stop(Seconds(simTime));

    sinkApps.Get(0)->TraceConnectWithoutContext("Rx", MakeCallback(&RxCallback));

    UdpClientHelper client(ueIpIface.GetAddress(0), port);
    client.SetAttribute("Interval", TimeValue(MilliSeconds(1)));
    client.SetAttribute("PacketSize", UintegerValue(1400));
    client.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));

    ApplicationContainer clientApps = client.Install(remoteHost);
    clientApps.Start(Seconds(1.5));
    clientApps.Stop(Seconds(simTime));

    // Attach initial
    lteHelper->Attach(ueDevs.Get(0), enbDevs.Get(0));

    lteHelper->EnableTraces();

    // Callbacks
    Config::Connect(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/LteUePhy/ReportCurrentCellRsrpSinr",
        MakeCallback(&RsrpSinrCallback));

    Config::Connect(
        "/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
        MakeCallback(&HandoverEnd));

    // CSV
    csvFile.open("lte-random-walk.csv");
    csvFile << "Time,X,Y,"
            << "RSRP_eNB1_dBm,Throughput_eNB1_Mbps,"
            << "RSRP_eNB2_dBm,Throughput_eNB2_Mbps,"
            << "Connected_eNB,Actual_Throughput_Mbps\n";

    Simulator::Schedule(Seconds(1.0), &LogData, ueNodes.Get(0));

    // NetAnim
    AnimationInterface anim("lte-random-walk.xml");
    anim.UpdateNodeDescription(enbNodes.Get(0), "eNB-1");
    anim.UpdateNodeDescription(enbNodes.Get(1), "eNB-2");
    anim.UpdateNodeDescription(ueNodes.Get(0), "UE");

    // Run
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    csvFile.close();
    Simulator::Destroy();

    return 0;
}
