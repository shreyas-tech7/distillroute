"""The synthetic domain: 600 small-business support tickets in 12 categories, written from scratch.

About 15% of the tickets are vague on purpose, so they carry no clue to their category.

Each category has its own sentence templates and slot words. A seeded generator fills them, so the
same seed always gives the same 600 tickets. Nothing here comes from a real customer.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

PER_CATEGORY = 50

SLOTS = {
    "item": ["blender", "desk lamp", "yoga mat", "coffee grinder", "phone case", "backpack", "kettle", "rug"],
    "days": ["three", "five", "seven", "ten", "twelve"],
    "amount": ["$18", "$42", "$75", "$120", "$260"],
    "city": ["Austin", "Denver", "Tulsa", "Omaha", "Reno", "Boise"],
    "plan": ["Starter", "Team", "Pro"],
    "page": ["checkout page", "login page", "product page", "cart", "account page"],
}

TEMPLATES = {
    "order_status": [
        "Where is my {item}? I ordered it {days} days ago.",
        "My order for a {item} still says processing after {days} days.",
        "Can you tell me when my {item} will ship?",
        "Has my {item} order left the warehouse yet?",
        "I have not heard anything about my {item} order since I paid.",
    ],
    "shipping_delay": [
        "Tracking says my {item} has been stuck in {city} for {days} days.",
        "The courier keeps moving the delivery date for my {item}.",
        "My package was due last week and it is still not here.",
        "The {item} shows out for delivery but nothing arrived in {city}.",
        "Why is shipping to {city} taking so long?",
    ],
    "return_request": [
        "I want to send the {item} back. How do I start a return?",
        "The {item} is not what I expected and I would like to return it.",
        "Can I return a {item} that I opened {days} days ago?",
        "Please send me a return label for my {item}.",
        "How long do I have to return the {item}?",
    ],
    "refund_status": [
        "I returned the {item} {days} days ago and still have no refund.",
        "When will the {amount} refund show up on my card?",
        "You said my refund was approved but I do not see {amount} yet.",
        "My refund for the {item} seems to be missing.",
        "Can you check on the refund for my returned {item}?",
    ],
    "damaged_item": [
        "My {item} arrived cracked.",
        "The box was crushed and the {item} inside is broken.",
        "The {item} I received has a dent on the side.",
        "Half of the {item} was missing parts when I opened it.",
        "The {item} stopped working on the first day.",
    ],
    "wrong_item": [
        "I ordered a {item} but you sent me something else.",
        "The package had the wrong color {item} in it.",
        "I got someone else's order instead of my {item}.",
        "You shipped two of the wrong {item} and none of the right one.",
        "This is not the {item} on my receipt.",
    ],
    "billing_error": [
        "I was charged {amount} twice for one order.",
        "My card shows a charge of {amount} that I do not recognize.",
        "The invoice says {amount} but the checkout said less.",
        "You billed me for the {plan} plan after I cancelled it.",
        "Why was I charged tax twice on my {item}?",
    ],
    "subscription_change": [
        "How do I move from the {plan} plan to a cheaper one?",
        "Please cancel my {plan} subscription at the end of this month.",
        "I want to pause my subscription for a few months.",
        "Can I upgrade to the {plan} plan halfway through the month?",
        "How do I turn off auto renewal on my {plan} plan?",
    ],
    "account_access": [
        "I cannot log in to my account.",
        "The password reset email never arrives.",
        "My account got locked after a few wrong passwords.",
        "I changed my email and now I cannot sign in.",
        "The two step code is not working for my account.",
    ],
    "website_bug": [
        "The {page} keeps showing an error.",
        "The {page} will not load on my phone.",
        "When I click pay on the {page} nothing happens.",
        "The {page} shows the wrong price for the {item}.",
        "Your {page} freezes every time I scroll.",
    ],
    "product_question": [
        "Does the {item} come with a warranty?",
        "Is the {item} safe to put in the dishwasher?",
        "What size is the {item} exactly?",
        "Will the {item} work with a European plug?",
        "Is the {item} back in stock soon?",
    ],
    "wholesale_inquiry": [
        "We run a shop in {city} and want to buy the {item} in bulk.",
        "Do you offer wholesale prices on orders over {amount}?",
        "Can our store in {city} become a reseller?",
        "What is the minimum order for wholesale {item} stock?",
        "We would like a quote for 200 units of the {item}.",
    ],
}

# Vague tickets read the same in any category. About 15% of each category draws one, so no
# student can reach 100%. Real queues have these too, and they are the cases a router should defer.
VAGUE = [
    "I need help with my {item} order.",
    "Something is wrong with my order.",
    "Can someone look at my account please?",
    "I have a problem with the {item}.",
    "Please call me about my {item}.",
    "My last order from {city} is a mess.",
]
VAGUE_SHARE = 0.15

OPENERS = ["", "", "Hi, ", "Hello. ", "Quick question. ", "Hey team, "]
CLOSERS = ["", "", " Thanks.", " Please help.", " Thank you!", " This is urgent."]

CATEGORIES = sorted(TEMPLATES)


def _fill(template: str, rng: random.Random) -> str:
    out = template
    for key, words in SLOTS.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(words), 1)
    return out


def generate(seed: int = 7) -> list[tuple[str, str]]:
    """Return 600 (text, category) pairs. Same seed, same tickets. Duplicate texts are redrawn."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cat in CATEGORIES:
        made = 0
        while made < PER_CATEGORY:
            pool = VAGUE if rng.random() < VAGUE_SHARE else TEMPLATES[cat]
            body = _fill(rng.choice(pool), rng)
            text = rng.choice(OPENERS) + body + rng.choice(CLOSERS)
            if text in seen:
                continue
            seen.add(text)
            rows.append((text, cat))
            made += 1
    rng.shuffle(rows)
    return rows


def write(path: Path, seed: int = 7) -> int:
    rows = generate(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["text", "category"])
        w.writerows(rows)
    return len(rows)
