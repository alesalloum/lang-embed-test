#!/usr/bin/env python3
"""Generate a claim–stance dataset on AI datacenter expansion.

Purpose
-------
Build exploratory data for testing whether text embeddings encode
**semantic stance** (supportive / neutral / critical) alongside topic
semantics. Each claim has three distinct social-media-style posts that
share the same underlying proposition but differ in attitude.

Schema mirrors the existing toy-posts layout:
  - Nested JSON: one object per claim, with texts.{supportive,critical,neutral}
  - Flat JSONL/CSV: one row per stance post (stance plays the role language
    played in the multilingual toy set)

Scale: 500 claims × 3 stances = 1500 text records (English).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TOPIC = "ai_datacenters"
TOPIC_LABEL = "Increasing number of AI datacenters"
LANGUAGE = "en"
LANGUAGE_NAME = "English"
N_CLAIMS = 500
SEED = 42

STANCES = {
    "supportive": {
        "label": "Supportive",
        "description": (
            "Agrees with or frames the claim positively; expresses "
            "enthusiasm, confidence, or optimism."
        ),
    },
    "critical": {
        "label": "Critical",
        "description": (
            "Disagrees with, refutes, or frames the claim negatively; "
            "expresses concern, skepticism, or opposition."
        ),
    },
    "neutral": {
        "label": "Neutral",
        "description": (
            "Expresses uncertainty, presents balanced/vague commentary, "
            "asks a question, or notes the situation without taking a side."
        ),
    },
}

ASPECTS = {
    "economic": {
        "label": "Economic",
        "description": "Jobs, investment, taxes, energy prices, regional growth, capital markets.",
    },
    "environmental": {
        "label": "Environmental",
        "description": "Energy use, water, emissions, land use, renewables, climate tradeoffs.",
    },
    "infrastructure": {
        "label": "Infrastructure",
        "description": "Power grids, transmission, cooling, fiber, roads, construction capacity.",
    },
    "geopolitical": {
        "label": "Geopolitical",
        "description": "National security, chip supply, sovereignty, alliances, export controls.",
    },
    "local_community": {
        "label": "Local community",
        "description": "Zoning, noise, housing, schools, community benefits, local politics.",
    },
    "technological": {
        "label": "Technological",
        "description": "GPU clusters, cooling tech, efficiency, networking, model training scale.",
    },
}

# ---------------------------------------------------------------------------
# Claim component banks (combinatorial generation with uniqueness checks)
# ---------------------------------------------------------------------------

PLACES = [
    "northern Virginia", "central Texas", "rural Iowa", "Arizona's desert corridor",
    "eastern Oregon", "upstate New York", "Georgia's coastal plain", "Ohio's Rust Belt towns",
    "Quebec's hydropower belt", "Ireland's midlands", "Singapore's industrial parks",
    "South Korea's semiconductor belt", "the UAE's free zones", "Chile's Atacama edge",
    "Finland's cold-climate parks", "Spain's solar belt", "Poland's logistics corridor",
    "Malaysia's Johor corridor", "India's Hyderabad tech belt", "Brazil's São Paulo hinterland",
    "Canada's prairie provinces", "the Nordics", "Gulf Coast industrial zones",
    "the Pacific Northwest", "Midwest farm counties", "suburban Phoenix",
    "northern Italy's industrial belt", "Japan's Tohoku rebuild zones",
    "Australia's renewable zones", "Morocco's Atlantic coast",
]

ACTORS = [
    "hyperscalers", "AI labs", "cloud providers", "chipmakers", "utility companies",
    "private equity funds", "state investment boards", "municipal governments",
    "regional economic-development agencies", "sovereign wealth funds",
    "colocation operators", "telecom carriers", "energy traders",
    "construction consortia", "national labs", "defense contractors",
    "startup GPU cloud firms", "REIT-backed campus developers",
]

FACILITIES = [
    "AI training campuses", "inference megacampuses", "GPU-dense halls",
    "liquid-cooled clusters", "multi-gigawatt campuses", "edge AI pods",
    "sovereign compute hubs", "research supercomputer halls",
    "modular container datacenters", "subsea-cable landing campuses",
    "nuclear-adjacent compute parks", "behind-the-meter solar campuses",
    "waste-heat district heating sites", "black-start resilient bunkers",
]

ECONOMIC_EFFECTS = [
    "create high-paying construction and facilities jobs",
    "inflate local residential electricity rates",
    "attract supplier ecosystems and vendor offices",
    "drive up industrial land prices near substations",
    "generate property-tax windfalls for school districts",
    "crowd out smaller manufacturers competing for power",
    "stimulate regional GDP through capital expenditure spikes",
    "concentrate wealth among a few campus operators",
    "require multi-billion-dollar grid upgrade subsidies",
    "produce temporary boom-bust hiring cycles",
    "increase commercial real-estate absorption rates",
    "shift municipal budgets toward infrastructure matching funds",
    "raise bond ratings for counties with long-term offtake deals",
    "divert scarce skilled trades from housing construction",
    "unlock exportable AI services as a new trade category",
]

ENVIRONMENTAL_EFFECTS = [
    "drive unprecedented electricity demand growth",
    "strain regional freshwater supplies for evaporative cooling",
    "accelerate utility renewable procurement timelines",
    "increase peak-load fossil peaker plant dispatch",
    "convert farmland and habitat into paved campuses",
    "enable waste-heat reuse for greenhouses and district heating",
    "push operators toward closed-loop liquid cooling",
    "raise Scope 2 emissions even when marketed as green",
    "compete with household electrification for clean electrons",
    "justify new nuclear and geothermal project pipelines",
    "create thermal plumes that alter local microclimates",
    "increase embodied carbon from concrete and steel builds",
    "incentivize 24/7 carbon-free energy matching contracts",
    "pressure aquifers during multi-year droughts",
    "displace quieter land uses with continuous industrial noise",
]

INFRA_EFFECTS = [
    "force multi-year transmission queue reforms",
    "require new high-voltage lines across rural counties",
    "overload aging substations during summer peaks",
    "spur dedicated behind-the-meter generation campuses",
    "consume fiber dark capacity on long-haul routes",
    "necessitate redundant water mains and chillers",
    "delay residential interconnection applications",
    "drive prefabricated modular power-block manufacturing",
    "create single points of failure on regional grids",
    "accelerate HVDC corridor planning between wind belts and campuses",
    "require hardened diesel and battery bridging for outages",
    "stretch transformer and switchgear supply chains",
    "push utilities into capacity auctions for firm power",
    "demand new rail spurs for heavy equipment delivery",
    "compete with EV chargers for feeder capacity",
]

GEO_EFFECTS = [
    "become a proxy contest for AI compute sovereignty",
    "reshape export-control debates around GPU clusters",
    "tie alliance politics to preferred hosting jurisdictions",
    "create strategic dependencies on foreign chip fabs",
    "motivate dual-use research security screening at campuses",
    "shift soft power toward countries with cheap clean power",
    "invite sanctions risk for cross-border training fleets",
    "drive friend-shoring of inference capacity",
    "elevate undersea cable security as national policy",
    "spark disputes over data residency and model weights",
    "make energy diplomacy inseparable from AI strategy",
    "encourage state-owned compute champions",
    "raise concerns about foreign ownership of critical campuses",
    "accelerate race dynamics for frontier-model training capacity",
    "blur lines between civilian AI labs and defense compute",
]

LOCAL_EFFECTS = [
    "trigger heated zoning fights at county board meetings",
    "promise community benefit agreements that residents distrust",
    "increase truck traffic and road wear near build sites",
    "offer scholarships and STEM labs as political sweeteners",
    "raise housing demand without matching school capacity",
    "spark noise complaints from 24/7 cooling systems",
    "reshape local identity from farming towns to tech corridors",
    "divide neighbors over tax abatements for billion-dollar firms",
    "create temporary hotel and RV-camp worker economies",
    "leave residents worried about water rights seniority",
    "fund emergency services while straining volunteer fire departments",
    "invite referendum campaigns against campus expansions",
    "change night skies with security lighting and flare stacks",
    "produce uneven job access for long-time residents",
    "make local elections hinge on datacenter permit votes",
]

TECH_EFFECTS = [
    "push liquid cooling from niche to default for dense racks",
    "require optical interconnect fabrics at campus scale",
    "drive custom ASIC deployment beyond commodity GPUs",
    "force new power-delivery designs inside halls",
    "raise the bar for cluster schedulers and fault isolation",
    "accelerate research into immersion cooling fluids",
    "create demand for higher-voltage rack architectures",
    "expose limits of air cooling for next-gen accelerators",
    "incentivize disaggregated memory and CXL fabrics",
    "make training runs sensitive to regional latency topologies",
    "promote on-site hydrogen and fuel-cell bridging experiments",
    "push software toward energy-aware job packing",
    "spawn markets for secondary GPU cloud capacity",
    "require tighter telemetry for thermal runaway prevention",
    "reshape MLOps around multi-campus failover",
]

CLAIM_FRAMES = [
    "The rapid buildout of {facilities} by {actors} in {place} will {effect}.",
    "As {actors} race to open more {facilities}, communities in {place} will {effect}.",
    "Expanding AI datacenter capacity across {place}, led by {actors}' {facilities}, is set to {effect}.",
    "The surge of new {facilities} backed by {actors} in {place} tends to {effect}.",
    "Growth of {facilities} from {actors} in {place} means the region will {effect}.",
    "With {actors} siting more {facilities} near {place}, the area will {effect}.",
    "The next wave of {facilities} that {actors} are stacking into {place} is widely expected to {effect}.",
    "AI datacenter expansion led by {actors} via {facilities} in {place} is already starting to {effect}.",
    "Planners argue that {actors} adding {facilities} around {place} will inevitably {effect}.",
    "Industry forecasts say {facilities} proliferating under {actors} in {place} will {effect}.",
]

# Standalone, non-combinatorial claim stems for extra diversity (hand-tuned angles)
HAND_CLAIMS: dict[str, list[str]] = {
    "economic": [
        "AI datacenter tax abatements often outlast the temporary construction payroll they advertise.",
        "Counties that host AI campuses see industrial electricity demand grow faster than residential ratepayer protections.",
        "GPU campus CapEx now rivals traditional auto-plant investments in several U.S. states.",
        "Energy hedges for AI campuses are becoming a material line item on utility balance sheets.",
        "Secondary markets for stranded GPU capacity are turning idle racks into speculative assets.",
        "AI datacenter boom towns risk Dutch-disease effects when one tenant dominates the tax base.",
        "Long-term power purchase agreements for AI training lock in prices that reshape wholesale markets.",
        "Local chambers of commerce treat AI campuses as 'anchor tenants' even when headcount stays lean.",
        "Insurance premiums for high-density compute halls are rising faster than for conventional warehouses.",
        "Municipal bond prospectuses increasingly cite AI datacenter pipelines as growth catalysts.",
        "Construction overtime for AI campuses is bidding electricians away from hospital projects.",
        "AI inference campuses create steadier revenue than training halls because utilization is continuous.",
        "Some governors pitch AI datacenters as replacements for shuttered coal-plant payrolls.",
        "Vendor ecosystems around coolant, busway, and rack vendors become regional export niches.",
        "Property assessors struggle to value custom AI halls that have few comparable sales.",
        "AI campus land banking near substations is pricing out logistics warehouses.",
        "State incentive packages for AI compute now compete directly with semiconductor fab bids.",
        "Retail electricity customers absorb transmission riders justified by AI load growth.",
        "Private credit funds are underwriting AI datacenter shells before tenants are signed.",
        "Economic-impact studies for AI campuses routinely undercount grid upgrade externalities.",
    ],
    "environmental": [
        "AI datacenter water withdrawals can exceed residential use in drought-stressed counties.",
        "Corporate 100% renewable claims for AI campuses often ignore hourly grid carbon intensity.",
        "New gas peakers are being permitted explicitly to firm AI datacenter loads.",
        "Closed-loop cooling is marketed as green while still raising facility energy intensity.",
        "Concrete pours for AI megacampuses lock in decades of embodied carbon before the first model trains.",
        "Wildlife corridors are being fragmented by AI campus security fencing and access roads.",
        "Waste-heat reuse pilots remain rare relative to the thermal energy AI halls reject.",
        "AI load growth is delaying coal retirements in several U.S. and European grids.",
        "Operators prefer arid sites for cheap land even when water scarcity is acute.",
        "24/7 carbon-free matching for AI campuses is still more marketing than measured practice.",
        "Noise from adiabatic coolers travels farther at night in sparsely populated valleys.",
        "PFAS concerns are emerging around some specialty immersion cooling fluids.",
        "Datacenter diesel backup fleets add localized NOx during grid emergencies.",
        "Land cleared for AI campuses rarely returns to agriculture after lease expiry scenarios.",
        "Hyperscalers' climate pledges are increasingly stress-tested by AI power curves.",
        "River-temperature permits constrain once-through cooling options near AI campuses.",
        "Solar-plus-storage campuses for AI still rely on grid imports during multi-day lulls.",
        "E-waste from rapid GPU refresh cycles is outpacing certified recycling capacity.",
        "Methane leakage from gas supply chains undermines 'bridge fuel' narratives for AI power.",
        "Environmental impact statements for AI campuses often treat cumulative regional builds lightly.",
    ],
    "infrastructure": [
        "Interconnection queues for AI campuses now dominate several regional transmission plans.",
        "Transformer lead times are a harder bottleneck for AI datacenters than land acquisition.",
        "Utilities are proposing AI-only substations to isolate residential feeders from campus spikes.",
        "Fiber diversity requirements for AI campuses are driving redundant long-haul builds.",
        "Modular power skids are being factory-built because on-site electrical labor is scarce.",
        "Some grids are creating special AI load tariffs with curtailable interruptibility clauses.",
        "HVDC proposals are being revived specifically to move wind power to AI campus corridors.",
        "Water utilities lack metering granularity to audit AI campus cooling withdrawals in real time.",
        "Rail capacity for oversized transformers is becoming a logistics constraint for campus builds.",
        "Behind-the-meter SMRs are pitched as companions to multi-gigawatt AI parks.",
        "Sync condensers and grid-forming inverters are being added to stabilize AI-heavy buses.",
        "Campus black-start plans now include GPU-hall sequencing to avoid inrush collapses.",
        "Municipal stormwater systems were not designed for the impervious cover of AI megasites.",
        "Telecom meet-me rooms adjacent to AI halls are becoming scarce premium real estate.",
        "Spare generation capacity margins shrink when multiple AI campuses energize in the same season.",
        "Utilities are rewriting load forecasting models because AI ramps look unlike historical industrials.",
        "Dual-fed campus designs double transmission right-of-way needs across farmland.",
        "Chiller plant redundancy standards for AI are exceeding those of hospitals in some designs.",
        "Prefabricated hall modules reduce schedule risk but still wait on switchgear deliveries.",
        "Regional reliability coordinators now run contingency studies centered on AI campus trips.",
    ],
    "geopolitical": [
        "Governments increasingly treat domestic AI datacenter capacity as strategic infrastructure.",
        "Export controls on advanced GPUs are redirecting AI campus investment toward friendly jurisdictions.",
        "Allied nations are negotiating shared sovereign compute pools to reduce U.S. hyperscaler dependence.",
        "Foreign ownership reviews now scrutinize AI campus land deals near military bases.",
        "Countries with surplus hydro are marketing themselves as AI training destinations.",
        "Model-weight residency rules are forcing duplicate inference campuses across borders.",
        "Undersea cable landing rights are being leveraged in AI capacity diplomacy.",
        "Sanctions scenarios include cutting power or connectivity to adversary-linked AI halls.",
        "National AI strategies now list gigawatts of compute as explicitly as they list talent visas.",
        "Chip foundry geography constrains where frontier training campuses can reliably expand.",
        "Defense agencies are reserving priority access clauses in civilian AI campus contracts.",
        "Data localization laws multiply the number of AI campuses needed for the same product.",
        "Rival blocs are racing to host open-source model training as a soft-power play.",
        "Critical minerals for cooling and power electronics enter AI campus security briefings.",
        "Cross-border latency corridors influence which alliances co-locate inference capacity.",
        "State subsidies for AI campuses are framed as national-security industrial policy.",
        "Espionage concerns rise when foreign contractors maintain AI campus building systems.",
        "Neutral countries market political risk hedges to AI labs seeking training havens.",
        "GPU smuggling narratives intensify scrutiny of secondary cloud campuses.",
        "Diplomats now track AI megawatt announcements the way they once tracked steel output.",
    ],
    "local_community": [
        "County hearings on AI campuses routinely overflow with residents worried about water and rates.",
        "Community benefit agreements for AI datacenters often emphasize scholarships over rate relief.",
        "Farm families near AI sites report well-drawdown fears during peak cooling months.",
        "Local officials trade multi-decade tax abatements for a few dozen permanent facility jobs.",
        "RV parks and extended-stay hotels fill with temporary trades during AI campus builds.",
        "School boards debate whether AI campus PILOT payments replace lost farmland taxes fairly.",
        "Neighborhood groups organize against nighttime noise from AI cooling yards.",
        "Indigenous water rights claims are colliding with AI campus permitting in several regions.",
        "Small businesses cheer construction spending then struggle when the boom crew leaves.",
        "Churches and civic halls become organizing hubs for anti-datacenter ballot initiatives.",
        "Residents question why AI campuses get expedited permits while housing projects stall.",
        "Volunteer fire departments seek specialized training for lithium and coolant incidents.",
        "Local newspapers frame AI campuses as either salvation or colonization of rural land.",
        "Youth sports leagues gain sponsorships from campus operators seeking goodwill.",
        "Property values near AI sites diverge: industrial adjacency premiums vs. livability discounts.",
        "Town identity messaging shifts from agricultural heritage to 'AI corridor' branding.",
        "Public comment periods for AI campuses are accused of being too short for meaningful input.",
        "Immigrants and long-time residents compete unevenly for the skilled trades AI builds demand.",
        "Light pollution from secure campuses alters stargazing culture in dark-sky communities.",
        "Mayors campaign on landing an AI campus as proof of economic relevance.",
    ],
    "technological": [
        "Direct-to-chip liquid cooling is becoming mandatory for next-gen AI accelerator densities.",
        "Optical circuit switching inside AI halls is moving from research demos to procurement.",
        "Power delivery at 48V and higher is rewriting rack and busbar standards for AI campuses.",
        "Cluster schedulers now optimize for energy price and carbon intensity, not only FLOPS.",
        "Immersion cooling vendors are racing to prove fluid longevity under continuous AI loads.",
        "Custom AI ASICs change thermal maps enough that older air-cooled halls become obsolete.",
        "Multi-campus training requires WAN fabrics that behave more like memory interconnects.",
        "Telemetry for rack-level thermal runaway is a new safety-critical software stack.",
        "GPU cloud spot markets depend on rapid reconfiguration of AI datacenter partitions.",
        "Software-defined power capping is used to keep AI halls inside utility curtailment envelopes.",
        "CXL and disaggregated memory designs aim to stretch scarce HBM across more accelerators.",
        "On-site microgrids for AI campuses experiment with fuel cells during grid stress events.",
        "Model checkpointing strategies are adapting to multi-region power availability windows.",
        "Robotic hot-aisle maintenance is pitched to cut human exposure in ultra-dense AI halls.",
        "Firmware supply-chain attestation is tightening for baseboard management on AI fleets.",
        "Heat-reuse APIs let nearby greenhouses subscribe to AI campus thermal output schedules.",
        "Accelerator refresh cycles of 18–24 months force continuous hall reconfiguration.",
        "Edge inference pods near users reduce some core campus growth but add many small sites.",
        "Failure domains in AI megacampuses are redesigned after correlated cooling outages.",
        "Energy-proportional networking gear is prioritized because fabric draw rivals compute draw.",
    ],
}

ASPECT_EFFECTS = {
    "economic": ECONOMIC_EFFECTS,
    "environmental": ENVIRONMENTAL_EFFECTS,
    "infrastructure": INFRA_EFFECTS,
    "geopolitical": GEO_EFFECTS,
    "local_community": LOCAL_EFFECTS,
    "technological": TECH_EFFECTS,
}

SUPPORTIVE_OPENERS = [
    "This tracks.",
    "Honestly, this matches what I'm seeing.",
    "Hard agree.",
    "Yes — and it's overdue that we say it.",
    "Calling it now: this claim is solid.",
    "Paying attention pays off here.",
    "If you're following the buildout, this should ring true.",
    "Finally some clarity on the datacenter wave.",
    "This is the accurate read.",
    "Naming the reality matters.",
]

CRITICAL_OPENERS = [
    "Hard disagree.",
    "This claim is overcooked.",
    "We're being sold a story here.",
    "Skeptical doesn't even cover it.",
    "This is the part that doesn't survive contact with evidence.",
    "Not buying it.",
    "Causal leap alert.",
    "Red flag city.",
    "People will regret treating this as settled.",
    "This take papers over too much.",
]

NEUTRAL_OPENERS = [
    "Still collecting receipts on this.",
    "Not sure where I land yet.",
    "Interesting claim — evidence is mixed.",
    "Genuinely unsure how this nets out.",
    "Seeing this talked about a lot.",
    "Open question for me.",
    "Could go either way depending on execution.",
    "Noting this without a hot take.",
    "Curious what the longitudinal data will show.",
    "Parking this as 'watch closely'.",
]

SUPPORTIVE_CLOSERS = [
    "Worth amplifying.",
    "The signal is clear.",
    "I'm here for naming it plainly.",
    "Believe people when they show you the pattern.",
    "Momentum in the discourse helps.",
    "This is the right framing.",
    "Bullish on facing facts.",
    "Count me aligned.",
]

CRITICAL_CLOSERS = [
    "We should hit pause.",
    "The externalities are screaming.",
    "Accountability first.",
    "Don't say we weren't warned.",
    "This needs brakes, not cheerleading.",
    "Bad bargain.",
    "Push back while you still can.",
    "The bill always comes due.",
]

NEUTRAL_CLOSERS = [
    "What's your read?",
    "Anyone tracking primary sources?",
    "Time will tell.",
    "Need better data before choosing a side.",
    "Watching the next few quarters.",
    "Both boosters and critics have pieces of it.",
    "Filing under unresolved.",
    "Curious to hear local perspectives.",
]


def _slug_aspect(aspect: str) -> str:
    return aspect


def _claim_id(idx: int) -> str:
    return f"{TOPIC}_{idx:03d}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _hash_key(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return h


def generate_claims(rng: random.Random) -> list[dict]:
    """Build exactly N_CLAIMS unique claim records with aspect tags.

    Combinatorial claims also keep slot metadata so stance posts can paraphrase
    the same proposition without pasting the claim verbatim.
    """
    claims: list[dict] = []
    seen: set[str] = set()

    aspect_cycle = list(ASPECTS.keys())
    for aspect in aspect_cycle:
        for claim_text in HAND_CLAIMS[aspect]:
            key = _normalize(claim_text)
            if key in seen:
                continue
            seen.add(key)
            claims.append({"aspect": aspect, "claim": claim_text, "slots": None})

    attempts = 0
    max_attempts = N_CLAIMS * 80
    while len(claims) < N_CLAIMS and attempts < max_attempts:
        attempts += 1
        counts = {a: 0 for a in aspect_cycle}
        for c in claims:
            counts[c["aspect"]] += 1
        aspect = min(aspect_cycle, key=lambda a: (counts[a], aspect_cycle.index(a)))

        frame = rng.choice(CLAIM_FRAMES)
        place = rng.choice(PLACES)
        actors = rng.choice(ACTORS)
        facilities = rng.choice(FACILITIES)
        effect = rng.choice(ASPECT_EFFECTS[aspect])
        slots = {
            "place": place,
            "actors": actors,
            "facilities": facilities,
            "effect": effect,
        }
        claim_text = frame.format(**slots)
        variant = rng.random()
        if variant < 0.22:
            claim_text = claim_text.replace(" will ", " is likely to ", 1)
        elif variant < 0.44:
            claim_text = claim_text.replace(" will ", " is poised to ", 1)

        key = _normalize(claim_text)
        if key in seen:
            continue
        seen.add(key)
        claims.append({"aspect": aspect, "claim": claim_text, "slots": slots})

    if len(claims) < N_CLAIMS:
        raise RuntimeError(f"Only generated {len(claims)} unique claims; need {N_CLAIMS}")

    rng.shuffle(claims)
    claims = claims[:N_CLAIMS]

    by_aspect: dict[str, list[dict]] = {a: [] for a in aspect_cycle}
    for c in claims:
        by_aspect[c["aspect"]].append(c)
    ordered: list[dict] = []
    pointers = {a: 0 for a in aspect_cycle}
    while len(ordered) < N_CLAIMS:
        for a in aspect_cycle:
            i = pointers[a]
            if i < len(by_aspect[a]):
                ordered.append(by_aspect[a][i])
                pointers[a] = i + 1
            if len(ordered) >= N_CLAIMS:
                break

    out: list[dict] = []
    for idx, c in enumerate(ordered, start=1):
        aspect = c["aspect"]
        row = {
            "claim_id": _claim_id(idx),
            "topic": TOPIC,
            "topic_label": TOPIC_LABEL,
            "aspect": aspect,
            "aspect_label": ASPECTS[aspect]["label"],
            "claim": c["claim"],
        }
        if c.get("slots"):
            row["_slots"] = c["slots"]
        out.append(row)
    return out


def _clean_post(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" .", ".").replace(" ,", ",")
    s = s.replace("?.", "?").replace("!.", "!")
    s = s.replace("— —", "—")
    s = re.sub(r"''+", "'", s)
    s = re.sub(r"\?\?", "?", s)
    return s


def _stance_from_slots(slots: dict, rng: random.Random) -> dict[str, str]:
    place = slots["place"]
    actors = slots["actors"]
    facilities = slots["facilities"]
    effect = slots["effect"]
    actors_cap = actors[0].upper() + actors[1:]

    supp = [
        (
            f"{rng.choice(SUPPORTIVE_OPENERS)} The {facilities} wave in {place} really will "
            f"{effect}. When {actors} move at this scale, that outcome is basically baked in. "
            f"{rng.choice(SUPPORTIVE_CLOSERS)}"
        ),
        (
            f"I'm aligned with the read that {actors}' push for more {facilities} around "
            f"{place} is set to {effect}. {rng.choice(SUPPORTIVE_CLOSERS)}"
        ),
        (
            f"{rng.choice(SUPPORTIVE_OPENERS)} Watching {place}: more {facilities} from "
            f"{actors} are going to {effect}. That's not hype — it's the pattern. "
            f"{rng.choice(SUPPORTIVE_CLOSERS)}"
        ),
        (
            f"Supportive take on {place}: {facilities} backed by {actors} "
            f"are going to {effect}. {rng.choice(SUPPORTIVE_CLOSERS)}"
        ),
    ]
    crit = [
        (
            f"{rng.choice(CRITICAL_OPENERS)} The idea that {facilities} in {place} must "
            f"{effect} is a stretch. {actors_cap} love deterministic narratives; reality is messier. "
            f"{rng.choice(CRITICAL_CLOSERS)}"
        ),
        (
            f"Pushing back: tying the {facilities} boom in {place} to '{effect}' oversells "
            f"what {actors} can actually force. {rng.choice(CRITICAL_CLOSERS)}"
        ),
        (
            f"{rng.choice(CRITICAL_OPENERS)} No, expanding {facilities} near {place} does not "
            f"automatically {effect}. That causal jump needs better proof. "
            f"{rng.choice(CRITICAL_CLOSERS)}"
        ),
        (
            f"Critical take: {actors} adding {facilities} in {place} is not guaranteed to "
            f"{effect}. {rng.choice(CRITICAL_CLOSERS)}"
        ),
    ]
    neut = [
        (
            f"{rng.choice(NEUTRAL_OPENERS)} People say {facilities} in {place} will "
            f"{effect}. Could be {actors}' planning docs, could be vibes. "
            f"{rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"Neutral note on {place}: if {actors} keep adding {facilities}, maybe that will "
            f"{effect} — or maybe local constraints dominate. {rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"{rng.choice(NEUTRAL_OPENERS)} Flagging the claim that more {facilities} around "
            f"{place} will {effect}. {rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"Just logging this: will {actors}' {facilities} in {place} really "
            f"{effect}? Unclear. {rng.choice(NEUTRAL_CLOSERS)}"
        ),
    ]
    return {
        "supportive": _clean_post(rng.choice(supp)),
        "critical": _clean_post(rng.choice(crit)),
        "neutral": _clean_post(rng.choice(neut)),
    }


def _stance_from_hand_claim(claim: str, rng: random.Random) -> dict[str, str]:
    """Paraphrase-oriented stance posts for hand-authored claims."""
    core = claim[:-1] if claim.endswith(".") else claim
    # Split into a short hook + rest for varied packaging
    words = core.split()
    if len(words) > 14:
        hook = " ".join(words[:8])
        rest = " ".join(words[8:])
    else:
        hook = core
        rest = ""

    supp = [
        (
            f"{rng.choice(SUPPORTIVE_OPENERS)} {core}. "
            f"That matches the buildout story I keep seeing. {rng.choice(SUPPORTIVE_CLOSERS)}"
        ),
        (
            f"I'm with this read: {core}. "
            f"{rng.choice(SUPPORTIVE_CLOSERS)}"
        ),
        (
            f"{rng.choice(SUPPORTIVE_OPENERS)} On '{hook}…' — yes, and the rest follows: "
            f"{rest or 'the pattern is consistent'}. {rng.choice(SUPPORTIVE_CLOSERS)}"
            if rest
            else (
                f"{rng.choice(SUPPORTIVE_OPENERS)} Affirming it plainly: {core}. "
                f"{rng.choice(SUPPORTIVE_CLOSERS)}"
            )
        ),
        (
            f"Supportive take: {core}. Confidence high. {rng.choice(SUPPORTIVE_CLOSERS)}"
        ),
    ]
    crit = [
        (
            f"{rng.choice(CRITICAL_OPENERS)} '{core}' sounds tidy and still isn't proven. "
            f"{rng.choice(CRITICAL_CLOSERS)}"
        ),
        (
            f"Pushing back on this one. {core}? That's a stronger claim than the evidence. "
            f"{rng.choice(CRITICAL_CLOSERS)}"
        ),
        (
            f"{rng.choice(CRITICAL_OPENERS)} Treating '{hook}…' as settled skips the hard "
            f"parts{(': ' + rest) if rest else ''}. {rng.choice(CRITICAL_CLOSERS)}"
        ),
        (
            f"Critical take: I don't accept that {core[0].lower() + core[1:]}. "
            f"{rng.choice(CRITICAL_CLOSERS)}"
        ),
    ]
    neut = [
        (
            f"{rng.choice(NEUTRAL_OPENERS)} Circulating claim: {core}. "
            f"{rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"Neutral note — neither buying nor burying it: {core}. "
            f"{rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"{rng.choice(NEUTRAL_OPENERS)} Re: '{hook}…'{(' / ' + rest) if rest else ''}. "
            f"Need sharper measurement. {rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"Just parking this without a verdict: {core}. {rng.choice(NEUTRAL_CLOSERS)}"
        ),
    ]
    # Filter out any empty/broken from conditional
    supp = [_clean_post(x) for x in supp if x and len(x) > 40]
    crit = [_clean_post(x) for x in crit if x and len(x) > 40]
    neut = [_clean_post(x) for x in neut if x and len(x) > 40]
    return {
        "supportive": rng.choice(supp),
        "critical": rng.choice(crit),
        "neutral": rng.choice(neut),
    }


def generate_stance_texts(claim_row: dict, rng: random.Random) -> dict[str, str]:
    """Create three distinct stance posts for the same underlying claim."""
    slots = claim_row.get("_slots")
    if slots:
        texts = _stance_from_slots(slots, rng)
    else:
        texts = _stance_from_hand_claim(claim_row["claim"], rng)

    if len(set(texts.values())) < 3:
        low = claim_row["claim"]
        low = low[0].lower() + low[1:] if low else low
        texts["critical"] = _clean_post(
            f"Hard disagree. The claim that {low.rstrip('.')} does not hold up. "
            f"{rng.choice(CRITICAL_CLOSERS)}"
        )
        texts["neutral"] = _clean_post(
            f"Noting without choosing a side: {claim_row['claim'].rstrip('.')}. "
            f"{rng.choice(NEUTRAL_CLOSERS)}"
        )
    return texts


def build_records(claims: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Return (nested claim objects, flat stance rows)."""
    nested: list[dict] = []
    flat: list[dict] = []
    for c in claims:
        texts = generate_stance_texts(c, rng)
        public = {k: v for k, v in c.items() if not k.startswith("_")}
        nested.append({**public, "texts": texts})
        for stance, text in texts.items():
            flat.append(
                {
                    "post_id": f"{c['claim_id']}_{stance}",
                    "claim_id": c["claim_id"],
                    "topic": c["topic"],
                    "topic_label": c["topic_label"],
                    "aspect": c["aspect"],
                    "aspect_label": c["aspect_label"],
                    "claim": c["claim"],
                    "stance": stance,
                    "language": LANGUAGE,
                    "language_name": LANGUAGE_NAME,
                    "text": text,
                }
            )
    return nested, flat


def write_outputs(nested_claims: list[dict], flat: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = DATA_DIR / "claims_stances"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flat JSONL
    jsonl_path = out_dir / "claims.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in flat:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Nested JSON
    aspect_counts = {a: 0 for a in ASPECTS}
    for c in nested_claims:
        aspect_counts[c["aspect"]] += 1

    nested_doc = {
        "description": (
            "Synthetic claim–stance social-media posts about the increasing number "
            "of AI datacenters, for testing whether embeddings capture semantic stance "
            "(supportive vs neutral vs critical) alongside topic semantics."
        ),
        "topic": TOPIC,
        "topic_label": TOPIC_LABEL,
        "aspects": ASPECTS,
        "stances": STANCES,
        "language": LANGUAGE,
        "language_name": LANGUAGE_NAME,
        "n_claims": len(nested_claims),
        "n_stances": len(STANCES),
        "n_records_flat": len(flat),
        "aspect_counts": aspect_counts,
        "claims": nested_claims,
    }
    json_path = out_dir / "claims.json"
    json_path.write_text(
        json.dumps(nested_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # CSV
    csv_path = out_dir / "claims.csv"
    fieldnames = [
        "post_id",
        "claim_id",
        "topic",
        "topic_label",
        "aspect",
        "aspect_label",
        "claim",
        "stance",
        "language",
        "language_name",
        "text",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in flat:
            writer.writerow(row)

    # Claims-only catalog (one row per claim, no stance texts)
    catalog_path = out_dir / "claim_catalog.jsonl"
    with catalog_path.open("w", encoding="utf-8") as f:
        for c in nested_claims:
            f.write(
                json.dumps(
                    {
                        "claim_id": c["claim_id"],
                        "topic": c["topic"],
                        "topic_label": c["topic_label"],
                        "aspect": c["aspect"],
                        "aspect_label": c["aspect_label"],
                        "claim": c["claim"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    meta = {
        "files": {
            "claims.json": "Nested: each claim has texts.{supportive,critical,neutral}",
            "claims.jsonl": "Flat records (one line per stance post)",
            "claims.csv": "Same flat schema as JSONL",
            "claim_catalog.jsonl": "One line per claim (no stance texts)",
        },
        "topic": TOPIC,
        "stances": list(STANCES.keys()),
        "aspects": list(ASPECTS.keys()),
        "language": LANGUAGE,
        "counts": {
            "claims": len(nested_claims),
            "stances_per_claim": 3,
            "total_text_records": len(flat),
            "aspect_counts": aspect_counts,
        },
        "embedding_notes": (
            "Ground-truth label for stance separation is `stance`. "
            "Each `claim_id` appears in 3 stance variants with the same underlying claim. "
            "A good embedding should separate supportive / neutral / critical while "
            "keeping same-claim posts nearer in topic space than unrelated claims."
        ),
    }

    readme = f"""# AI datacenter claim–stance posts

Synthetic social-media posts for studying whether embeddings capture **stance**
(supportive / neutral / critical) on a shared topic: **{TOPIC_LABEL}**.

## Topic

- `{TOPIC}`: {TOPIC_LABEL}

## Aspects

{chr(10).join(f"- `{k}`: {v['label']} — {v['description']}" for k, v in ASPECTS.items())}

## Stances

{chr(10).join(f"- `{k}`: {v['description']}" for k, v in STANCES.items())}

## Scale

- **{len(nested_claims)}** claims × **3** stances = **{len(flat)}** text records
- English only (`language=en`)
- Aspect mix: {", ".join(f"{k}={v}" for k, v in aspect_counts.items())}

## Files

| File | Format |
| --- | --- |
| `claims.json` | Nested: each claim has `texts.{{supportive,critical,neutral}}` |
| `claims.jsonl` | Flat: one JSON object per stance post |
| `claims.csv` | Flat CSV with the same columns |
| `claim_catalog.jsonl` | One object per claim (claim text only) |

## How it was built

```bash
python3 scripts/generate_ai_datacenter_claims.py
```

Claims combine hand-authored stems with combinatorial frames across places,
actors, facilities, and aspect-specific effects. For each claim, three distinct
social-media-style posts are generated (supportive / critical / neutral).

## Flat schema (key fields)

- `claim_id` — shared across the three stance posts
- `post_id` — `{{claim_id}}_{{stance}}`
- `aspect` / `aspect_label` — topical facet within AI datacenters
- `claim` — underlying proposition
- `stance` — `supportive` | `critical` | `neutral`
- `text` — social-media post expressing that stance toward the claim

## Intended use

Embed all {len(flat)} texts and evaluate whether vectors separate by `stance`
while still reflecting shared `claim_id` / `aspect` semantics — exploratory
tests for stance-aware embedding quality.

```json
{json.dumps(meta, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {jsonl_path} ({len(flat)} rows)")
    print(f"Wrote {json_path} ({len(nested_claims)} claims)")
    print(f"Wrote {csv_path}")
    print(f"Wrote {catalog_path}")
    print(f"Wrote {out_dir / 'README.md'}")
    print(f"Aspect counts: {aspect_counts}")


def validate(nested_claims: list[dict], flat: list[dict]) -> None:
    assert len(nested_claims) == N_CLAIMS, len(nested_claims)
    assert len(flat) == N_CLAIMS * 3, len(flat)
    claim_texts = [c["claim"] for c in nested_claims]
    assert len(set(_normalize(t) for t in claim_texts)) == N_CLAIMS, "duplicate claims"
    for c in nested_claims:
        assert set(c["texts"]) == set(STANCES)
        assert len(set(c["texts"].values())) == 3, c["claim_id"]
        for stance, text in c["texts"].items():
            assert len(text) >= 40, (c["claim_id"], stance, text)
    post_ids = [r["post_id"] for r in flat]
    assert len(set(post_ids)) == len(post_ids)
    print("Validation OK.")


def patch_data_readme() -> None:
    path = DATA_DIR / "README.md"
    marker = "## AI datacenter claim–stance set"
    section = """## AI datacenter claim–stance set

Claim–stance posts for **stance embedding** tests (supportive / neutral / critical)
on a single topic: increasing AI datacenters.

See [`claims_stances/`](claims_stances/) (`claims.json` / `claims.jsonl` / `claims.csv`).

Regenerate with:

```bash
python3 scripts/generate_ai_datacenter_claims.py
```

- **500** claims × **3** stances = **1500** English posts
- Aspects: economic, environmental, infrastructure, geopolitical, local community, technological
- Flat ground-truth label for stance separation: `stance`; shared meaning key: `claim_id`
"""
    text = path.read_text(encoding="utf-8")
    if marker in text:
        # Replace existing section through end or next ## if we re-run
        start = text.index(marker)
        text = text[:start].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text + "\n", encoding="utf-8")


def patch_root_readme() -> None:
    path = ROOT / "README.md"
    bullet = (
        "- **AI datacenter claim–stance set**: 500 claims × 3 stances "
        "(supportive/neutral/critical) under [`data/claims_stances/`](data/claims_stances/)\n"
    )
    text = path.read_text(encoding="utf-8")
    if "claims_stances" in text:
        return
    # Insert after the user-posts bullet in Data section
    anchor = "  topic distributions) and **100 posts each** (**100_000** user posts)\n"
    if anchor in text:
        text = text.replace(anchor, anchor + bullet)
    else:
        text = text.rstrip() + "\n\n" + bullet
    # Add regenerate command block near other regenerate instructions
    regen = (
        "\nRegenerate AI datacenter claim–stance posts with:\n\n"
        "```bash\n"
        "python3 scripts/generate_ai_datacenter_claims.py\n"
        "```\n"
    )
    if "generate_ai_datacenter_claims.py" not in text:
        insert_after = (
            "python3 scripts/generate_english_users.py\n```\n"
        )
        if insert_after in text:
            text = text.replace(insert_after, insert_after + regen)
        else:
            text = text.rstrip() + "\n" + regen
    path.write_text(text, encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    claims = generate_claims(rng)
    nested, flat = build_records(claims, rng)
    validate(nested, flat)
    write_outputs(nested, flat)
    patch_data_readme()
    patch_root_readme()
    print("Done.")


if __name__ == "__main__":
    main()
