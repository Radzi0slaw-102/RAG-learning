UNANSWERABLE_CVES = [
    {
        "cve_id": "CVE-2021-44228",
        "hint": "the remote code execution vulnerability in Apache Log4j disclosed in December 2021, widely known as Log4Shell",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2017-0144",
        "hint": "the SMB remote code execution vulnerability in Microsoft Windows exploited by the EternalBlue tool and the WannaCry ransomware outbreak",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2014-0160",
        "hint": "the OpenSSL heartbeat extension buffer over-read vulnerability disclosed in 2014, widely known as Heartbleed",
        "related_technique_id": None,
    },
    {
        "cve_id": "CVE-2019-0708",
        "hint": "the Remote Desktop Services remote code execution vulnerability in Windows disclosed in 2019, widely known as BlueKeep",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2020-0796",
        "hint": "the Microsoft Windows SMBv3 remote code execution vulnerability disclosed in 2020, widely known as SMBGhost",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2022-30190",
        "hint": "the Microsoft Support Diagnostic Tool remote code execution vulnerability disclosed in 2022, widely known as Follina",
        "related_technique_id": "T1204",
    },
    {
        "cve_id": "CVE-2023-23397",
        "hint": "the Microsoft Outlook elevation of privilege vulnerability exploited to leak NTLM hashes, disclosed in March 2023",
        "related_technique_id": "T1203",
    },
    {
        "cve_id": "CVE-2023-4966",
        "hint": "the Citrix NetScaler ADC and Gateway sensitive information disclosure vulnerability disclosed in 2023, known as Citrix Bleed",
        "related_technique_id": None,
    },
    {
        "cve_id": "CVE-2024-21762",
        "hint": "the Fortinet FortiOS out-of-bounds write vulnerability allowing unauthenticated remote code execution disclosed in 2024",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2020-0601",
        "hint": "the Windows CryptoAPI spoofing vulnerability affecting ECC certificate validation disclosed in 2020, known as CurveBall",
        "related_technique_id": "T1036",
    },
    {
        "cve_id": "CVE-2022-22965",
        "hint": "the Remote Code Execution vulnerability in Spring Framework disclosed in March 2022, widely known as Spring4Shell",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2023-28252",
        "hint": "the Windows Common Log File System Driver privilege escalation vulnerability actively exploited zero-day in 2023",
        "related_technique_id": None,
    },
    {
        "cve_id": "CVE-2023-38831",
        "hint": "the WinRAR remote code execution vulnerability triggered via malicious archive files, disclosed in August 2023",
        "related_technique_id": "T1204",
    },
    {
        "cve_id": "CVE-2023-20198",
        "hint": "the Cisco IOS XE Web UI privilege escalation vulnerability actively exploited as zero-day in October 2023",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2024-1709",
        "hint": "the ConnectWise ScreenConnect authentication bypass vulnerability disclosed in February 2024",
        "related_technique_id": "T1078",
    },
    {
        "cve_id": "CVE-2021-36934",
        "hint": "the Windows SAM database improper access control vulnerability disclosed in July 2021, widely known as HiveNightmare",
        "related_technique_id": None,
    },
    {
        "cve_id": "CVE-2018-8174",
        "hint": "the Internet Explorer VBScript engine remote code execution vulnerability disclosed in May 2018, known as Double Kill",
        "related_technique_id": "T1203",
    },
    {
        "cve_id": "CVE-2020-1350",
        "hint": "the Windows Domain Name System Server remote code execution vulnerability disclosed in 2020, widely known as SIGRed",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2021-3156",
        "hint": "the Sudo heap-based buffer overflow vulnerability affecting Unix-like systems disclosed in 2021, known as Baron Samedit",
        "related_technique_id": None,
    },
    {
        "cve_id": "CVE-2021-40444",
        "hint": "the MSHTML remote code execution vulnerability in Microsoft Office documents disclosed in September 2021",
        "related_technique_id": "T1204",
    },
    {
        "cve_id": "CVE-2022-26923",
        "hint": "the Active Directory Domain Services privilege escalation vulnerability via machine account certificates disclosed in 2022",
        "related_technique_id": "T1078",
    },
    {
        "cve_id": "CVE-2023-24932",
        "hint": "the Windows Secure Boot bypass vulnerability exploited by BlackLotus UEFI bootkit disclosed in May 2023",
        "related_technique_id": "T1542",
    },
    {
        "cve_id": "CVE-2024-30078",
        "hint": "the Windows Wi-Fi Driver remote code execution vulnerability disclosed in June 2024",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2017-0199",
        "hint": "the Microsoft Office OLE arbitrary code execution vulnerability exploited in phishing campaigns disclosed in 2017",
        "related_technique_id": "T1566",
    },
    {
        "cve_id": "CVE-2019-11707",
        "hint": "the Mozilla Firefox type confusion vulnerability in Array.pop actively exploited in 2019",
        "related_technique_id": "T1204",
    },
    {
        "cve_id": "CVE-2020-16898",
        "hint": "the Windows TCP/IP stack ICMPv6 router advertisement vulnerability disclosed in 2020, known as Bad Neighbor",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2022-41040",
        "hint": "the Microsoft Exchange Server server-side request forgery vulnerability disclosed in September 2022, known as ProxyNotShell",
        "related_technique_id": "T1210",
    },
    {
        "cve_id": "CVE-2024-21413",
        "hint": "the Microsoft Outlook remote code execution vulnerability bypassing Protected View disclosed in February 2024, known as MonikerLink",
        "related_technique_id": "T1204",
    },
]

UNANSWERABLE_TECHNIQUES = [
    {
        "technique_id": "T1566",
        "hint": "the ATT&CK technique covering phishing as an initial access method",
    },
    {
        "technique_id": "T1059",
        "hint": "the ATT&CK technique covering execution via command and scripting interpreters",
    },
    {
        "technique_id": "T1210",
        "hint": "the ATT&CK technique covering exploitation of remote services to gain access or spread across network hosts",
    },
    {
        "technique_id": "T1204",
        "hint": "the ATT&CK technique covering user execution, such as clicking malicious links or opening malicious attachments",
    },
    {
        "technique_id": "T1555",
        "hint": "the ATT&CK technique covering credentials from password stores and credential managers",
    },
    {
        "technique_id": "T1036",
        "hint": "the ATT&CK technique covering masquerading to disguise malicious utility or file identity",
    },
    {
        "technique_id": "T1078",
        "hint": "the ATT&CK technique covering valid accounts used to maintain access or bypass access controls",
    },
    {
        "technique_id": "T1542",
        "hint": "the ATT&CK technique covering pre-OS boot initialization modification to hide persistence and maintain control",
    },
    {
        "technique_id": "T1574",
        "hint": "the ATT&CK technique covering search order hijacking and DLL side-loading to execute payload",
    },
    {
        "technique_id": "T1055",
        "hint": "the ATT&CK technique covering process injection to evade detection and execute arbitrary code in legitimate processes",
    },
    {
        "technique_id": "T1020",
        "hint": "the ATT&CK technique covering automated exfiltration of sensitive data across a network",
    },
    {
        "technique_id": "T1562",
        "hint": "the ATT&CK technique covering impairment of defenses such as disabling security software or logging",
    },
    {
        "technique_id": "T1112",
        "hint": "the ATT&CK technique covering modification of system registry settings to hide configuration or maintain persistence",
    },
    {
        "technique_id": "T1071",
        "hint": "the ATT&CK technique covering application layer protocols used for command and control communications",
    },
    {
        "technique_id": "T1195",
        "hint": "the ATT&CK technique covering supply chain compromise of software dependencies or hardware supply",
    },
    {
        "technique_id": "T1569",
        "hint": "the ATT&CK technique covering system services execution via service control manager or system daemons",
    },
    {
        "technique_id": "T1056",
        "hint": "the ATT&CK technique covering input capture including keylogging and credential harvesting",
    },
    {
        "technique_id": "T1558",
        "hint": "the ATT&CK technique covering Kerberos ticket requests abuse such as Kerberoasting or AS-REP Roasting",
    },
    {
        "technique_id": "T1041",
        "hint": "the ATT&CK technique covering data exfiltration over C2 channel",
    },
    {
        "technique_id": "T1498",
        "hint": "the ATT&CK technique covering network denial of service attacks to disrupt infrastructure accessibility",
    },
    {
        "technique_id": "T1547",
        "hint": "the ATT&CK technique covering boot or logon autostart execution for persistent payload launching",
    },
    {
        "technique_id": "T1543",
        "hint": "the ATT&CK technique covering creation or modification of system processes like Windows or OS X services",
    },
    {
        "technique_id": "T1113",
        "hint": "the ATT&CK technique covering screen capture to gather information and sensitive user activity",
    },
    {
        "technique_id": "T1001",
        "hint": "the ATT&CK technique covering data obfuscation to hide C2 communication channels",
    },
    {
        "technique_id": "T1102",
        "hint": "the ATT&CK technique covering web service usage for C2 communications or data hosting",
    },
    {
        "technique_id": "T1567",
        "hint": "the ATT&CK technique covering exfiltration over web service to bypass traditional egress controls",
    },
    {
        "technique_id": "T1203",
        "hint": "the ATT&CK technique covering exploitation for client execution through malicious media or documents",
    },
]