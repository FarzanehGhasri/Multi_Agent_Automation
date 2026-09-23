---
document_id: SOP-IT-014
version: 3.1
process: VPN access provisioning for new personnel
aliases: vpn access, vpn provisioning, remote access for new employees, new joiner vpn, تخصیص دسترسی vpn به پرسنل جدید
sla_hours: 2
owner: Infrastructure & Security Operations
approved_by: Systems & Methods Committee
effective_date: 2024-05-12
---

# SOP-IT-014 - VPN Access Provisioning for New Personnel

## 1. Purpose and scope
This procedure governs the granting of corporate VPN (remote access) rights to a
newly hired employee or contractor, from the moment the onboarding ticket is
raised until the requester can authenticate against the corporate gateway.
It applies to every business unit and to all standard remote-access profiles.

## 2. Service level agreement
The end-to-end service level agreement (SLA) for this process is **2 hours**
during business hours. The SLA starts when the onboarding ticket reaches the
ServiceDesk queue and stops when the access confirmation is sent to the
requester. Ninety percent of requests are expected to complete inside the SLA.

## 3. Approved steps

| # | Step | Owner | Target (h) | Type | Mandatory control |
|---|------|-------|-----------|------|-------------------|
| 1 | Request registration | ServiceDesk intake bot | 0.1 | automated | no |
| 2 | Direct manager approval | Direct Manager | 0.5 | manual_approval | no |
| 3 | Security policy check | Security Operations rule engine | 0.4 | automated | no |
| 4 | Firewall rule provisioning | Network Operations | 0.8 | manual_provisioning | no |
| 5 | User notification and handover | ServiceDesk | 0.2 | automated | no |

## 4. Policy notes
- The direct manager's approval records business need; it may be delegated to a
  named deputy but never skipped.
- Every granted profile must be time-boxed and re-certified every 180 days.
- Standard remote-access profiles are pre-approved by Information Security, so
  no individual security sign-off is required for this process.
- All access decisions are written to the immutable audit log.
