# Acme Master Services Agreement (2024)

## 1. Parties and Term

This Master Services Agreement ("Agreement") is entered into between Acme Corp
("Provider") and the Customer. The initial term is twelve (12) months from the
Effective Date and renews automatically for successive twelve-month periods
unless either party gives sixty (60) days written notice of non-renewal.

## 2. Payment Terms

Payment is due within thirty (30) days of receipt of a valid invoice. Invoices
not disputed within fifteen (15) days are deemed accepted. Late amounts accrue
interest at 1.5% per month. All fees are exclusive of applicable taxes.

## 3. Service Levels

Provider will maintain 99.9% monthly uptime for Tier 1 services and 99.5% for
Tier 2 services, measured over a calendar month excluding scheduled maintenance.
Service credits apply when uptime falls below target, as described in Schedule B.

## 4. Support and Escalation

Support requests are triaged by severity. Sev 1 incidents receive a response
within one hour; Sev 2 within four business hours. Unresolved Sev 1 incidents
escalate to the on-call engineering lead after two hours, and to the VP of
Engineering after four hours.

## 5. Troubleshooting Reference

The following error codes may appear in the ingestion pipeline. Customers should
consult this table before opening a support ticket.

| Error Code | Severity | Meaning                          | Remedy                                  |
|------------|----------|----------------------------------|-----------------------------------------|
| E-4470     | 2        | Connector auth token expired     | Rotate the token in the admin console   |
| E-4471     | 3        | Ingestion worker backlog exceeded| Restart the ingestion worker            |
| E-4472     | 1        | Checkpoint store unreachable     | Page the on-call; do not restart        |
| E-4473     | 2        | Schema mismatch on inbound event | Re-run the OCSF normaliser              |

## 6. Confidentiality

Each party will protect the other's Confidential Information with the same care
it uses for its own, and no less than reasonable care. Confidential Information
does not include information that becomes public through no fault of the
receiving party.

## 7. Limitation of Liability

Neither party's aggregate liability will exceed the fees paid in the twelve
months preceding the claim. Neither party is liable for indirect, incidental, or
consequential damages, even if advised of their possibility.
