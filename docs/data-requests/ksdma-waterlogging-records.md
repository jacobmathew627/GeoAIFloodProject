# Data request: KSDMA waterlogging incident records

**Status: sent 2026-08-27.** Follow up once after two weeks (by
2026-09-10) if there is no reply; see the RTI fallback below if that
follow-up also goes unanswered.

**Why this matters more than any other request in this project.** The model
predicts flood *inundation* well (spatial-block AUC 0.824) but cannot predict
urban *waterlogging*, because no free dataset records it. Of 14 locations
documented in public reporting as recurrent Kochi waterlogging points, neither
the Sentinel-1 inventory nor NDEM — the national disaster-management
inventory, taken on the correct day — covers a single one. Satellites do not
see 20 cm of water in a street between buildings. Incident records are the only
source.

Open data has since been tested a third way and narrowed the gap without
closing it: OpenStreetMap supplies the district's drain, ditch and canal
network, which turns out to be a real predictor — proximity to a mapped
channel ranks 5th of 16 conditioning factors. But it predicts *where flooding
is plausible*, not *where waterlogging was observed*. Only incident records do
that.

Fill in the bracketed fields before sending.

---

## Email

**To:** KSDMA — State Emergency Operations Centre
`keralasdma@gmail.com`, `sec.disastermgmt@kerala.gov.in`
(Confirm current addresses at <https://sdma.kerala.gov.in> → Contact)

**Cc:** District Emergency Operations Centre, Ernakulam (via the District
Collectorate) — they hold the district-level 1077 logs.

**Subject:** Request for flood/waterlogging incident records — Ernakulam
district — academic flood modelling

---

Respected Sir/Madam,

I am [YOUR NAME], [YOUR ROLE, e.g. final-year B.Tech / M.Tech student] at
[YOUR INSTITUTION]. I am developing an open flood risk model for Ernakulam
district that combines terrain, land cover and IMD rainfall data to map where
flooding is most likely at a given rainfall depth.

The model performs well for river and backwater inundation, which satellite
data can observe. It cannot currently address **urban waterlogging**, because
street-level flooding is largely invisible to satellites — it drains before the
next overpass, and is hidden by tree canopy and buildings. The only reliable
record of where waterlogging actually occurs is the incident data held by the
disaster management authorities.

I would be grateful for access to the following, for Ernakulam district:

1. **Flood and waterlogging incident records**, 2018 to present, as logged
   through the 1077 helpline or the District Emergency Operations Centre —
   ideally with location (address, landmark, ward, or coordinates), date and
   time, and the incident category.

2. Any **list of recurrently waterlogged locations** maintained for
   monsoon preparedness, including the pre-monsoon inspection lists.

3. Relevant annexures of the **Ernakulam District Disaster Management Plan**
   identifying flood-prone locations, if these can be shared.

**On privacy:** I do not need any personal information. Caller names, phone
numbers and any other identifying details can be removed before sharing. If
point locations cannot be released, **counts aggregated to ward or local-body
level would still be very useful** — that is enough to test and calibrate the
model.

Any convenient format is fine (Excel, CSV, shapefile, or even scanned
registers).

**What I will do with it.** The work is non-commercial and academic. I will
acknowledge KSDMA as the data source in all outputs, and I am happy to share
the resulting waterlogging risk maps, the validation results, and the full
source code with KSDMA and with the Ernakulam district administration at no
cost. If it is useful, I can provide the maps in a format your GIS team can
open directly.

I am glad to complete any formal request procedure, provide a letter from my
institution, or meet in person if that would help.

Thank you for your time and for the work KSDMA does.

Respectfully,

[YOUR NAME]
[YOUR ROLE], [YOUR INSTITUTION]
[PHONE] · [EMAIL]

---

## Practical notes

**Attach or offer** a one-page summary with a sample map. A concrete artefact
makes the request read as real work rather than a speculative ask.

**Expect to be redirected.** The district EOC in Ernakulam, or Kochi
Corporation's health/engineering wing, may be the actual custodian. Ask
politely for the correct office rather than re-sending the same mail.

**Follow up once** after two weeks, briefly.

### If there is no reply: file an RTI

This data is held by a public authority, so the Right to Information Act 2005
applies. An RTI is a formal request that must be answered within 30 days, costs
**₹10**, and does not depend on goodwill.

- File online at <https://rtionline.gov.in> (Kerala state authorities are
  also reachable via <https://keralartiportal.kerala.gov.in>)
- Address it to the **Public Information Officer, KSDMA** (or PIO, District
  Collectorate Ernakulam for district logs)
- Keep the wording narrow and factual. Broad requests get refused as
  "disproportionate diversion of resources". For example:

  > Please provide the number of flood/waterlogging complaints received for
  > Ernakulam district through the 1077 helpline for the period 1 June 2018 to
  > 31 December 2025, with the location (ward or local body) and date of each,
  > in electronic form. Personal details of complainants are not required and
  > may be withheld.

- Ask explicitly for **electronic form** — otherwise you may receive
  photocopies.

Pre-empting the privacy objection and naming ward-level aggregation as
acceptable is what usually turns a refusal into a partial release.
