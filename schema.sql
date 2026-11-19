CREATE TABLE IF NOT EXISTS phones (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source_url TEXT NOT NULL UNIQUE,
    scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS specifications (
    phone_id BIGINT PRIMARY KEY REFERENCES phones(id) ON DELETE CASCADE,
    display_size TEXT,
    weight TEXT,
    dimensions TEXT,
    camera TEXT,
    battery TEXT,
    chipset TEXT,
    os TEXT,
    storage TEXT,
    raw_specs JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS prices (
    id BIGSERIAL PRIMARY KEY,
    phone_id BIGINT NOT NULL REFERENCES phones(id) ON DELETE CASCADE,
    price_text TEXT NOT NULL,
    currency TEXT,
    amount NUMERIC(12, 2),
    price_type TEXT NOT NULL DEFAULT 'GSMArena listed price',
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (phone_id, price_text)
);

-- Remove the obsolete column from databases created with an earlier schema.
ALTER TABLE specifications DROP COLUMN IF EXISTS network_technology;
