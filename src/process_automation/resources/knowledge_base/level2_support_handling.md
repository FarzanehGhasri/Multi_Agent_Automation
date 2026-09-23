---
document_id: SOP-IT-008
version: 5.0
process: Level 2 support ticket handling
aliases: level two support, l2 support, second line support, فرآیند پشتیبانی سطح ۲
sla_hours: 4
owner: IT Service Management
approved_by: Systems & Methods Committee
effective_date: 2024-02-18
---

# SOP-IT-008 - Level 2 Support Ticket Handling

## 1. Purpose and scope
This procedure covers incidents escalated from the first line to the second
line of support, from the moment the escalation is accepted until the customer
confirms that the service is restored.

## 2. Service level agreement
The end-to-end SLA for a level 2 incident of standard priority is **4 hours**.
Critical incidents follow the major-incident procedure (SOP-IT-009) instead.

## 3. Approved steps

| # | Step | Owner | Target (h) | Type | Mandatory control |
|---|------|-------|-----------|------|-------------------|
| 1 | Ticket triage and assignment | Service Desk dispatcher | 0.5 | waiting_queue | no |
| 2 | Level 1 diagnosis handover | Level 1 Support | 0.5 | data_entry | no |
| 3 | Level 2 investigation | Level 2 Engineer | 2.0 | manual_provisioning | no |
| 4 | Configuration change in CMDB | Level 2 Engineer | 0.5 | data_entry | no |
| 5 | Customer confirmation | Service Desk | 0.5 | automated | no |

## 4. Policy notes
- An escalated ticket must carry a named owner within 30 minutes of arriving in
  the level 2 queue.
- Every configuration change must be reflected in the CMDB before the ticket is
  closed.
- Repeat incidents on the same configuration item trigger a problem record.
