---
document_id: SOP-SEC-021
version: 2.4
process: Privileged database access provisioning
aliases: sensitive database access, privileged db access, production database rights, دسترسی حساس دیتابیس
sla_hours: 8
owner: Information Security Office
approved_by: Systems & Methods Committee
effective_date: 2024-09-03
---

# SOP-SEC-021 - Privileged Access to Production Databases

## 1. Purpose and scope
This procedure governs elevated (read-write, DDL or DBA) access to production
database instances holding customer or financial data. It complements, and does
not replace, the standard application-role request procedure.

## 2. Service level agreement
The end-to-end SLA is **8 hours** during business hours. The SLA is measured
from ticket submission to the moment the privilege is active in the target
instance.

## 3. Approved steps

| # | Step | Owner | Target (h) | Type | Mandatory control |
|---|------|-------|-----------|------|-------------------|
| 1 | Request registration | ServiceDesk intake bot | 0.2 | automated | no |
| 2 | Direct manager approval | Direct Manager | 1.0 | manual_approval | no |
| 3 | Security officer review | Information Security Officer | 4.0 | security_review | yes (ISMS-POL-07) |
| 4 | DBA privilege grant | Database Administration | 2.0 | manual_provisioning | no |
| 5 | Audit logging and notification | Compliance | 0.8 | automated | no |

## 4. Policy notes
- Clause ISMS-POL-07 of the information-security policy makes the security
  officer review a mandatory control for every privileged database grant. The
  review may be accelerated or supported by tooling, but it cannot be removed,
  delegated to an automated rule engine, or executed after the fact.
- Every privileged grant is time-boxed to a maximum of 72 hours and is revoked
  automatically on expiry.
- Segregation of duties: the requester, the approver and the grantor must be
  three different people.
- The security officer must record a written justification for every decision.
