"""SQLAlchemy ORM models matching `postgres/init.sql`.

The 96 Q-columns are declared programmatically to avoid 96 hand-typed lines.
"""

from datetime import date, datetime

from geoalchemy2 import Geography
from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Integer, Numeric, SmallInteger,
    String, Text, Float, Index, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ── POD METADATA ─────────────────────────────────────────────────────────────
class PodMetadata(Base):
    __tablename__ = "pod_metadata"

    pod:           Mapped[str]            = mapped_column(String(30), primary_key=True)
    d_dtall:       Mapped[date | None]    = mapped_column(Date)
    d_dtsmo:       Mapped[date | None]    = mapped_column(Date)
    d_flg1:        Mapped[float | None]   = mapped_column(Numeric)
    d_pot1:        Mapped[float | None]   = mapped_column(Numeric)
    d_potma:       Mapped[float | None]   = mapped_column(Numeric)
    d_potc:        Mapped[int | None]     = mapped_column(Integer)
    d_flatt:       Mapped[bool | None]    = mapped_column(Boolean)
    d_8potim:      Mapped[float | None]   = mapped_column(Numeric)
    d_tipta:       Mapped[str | None]     = mapped_column(String(20))
    d_49des:       Mapped[str | None]     = mapped_column(Text)
    d_viafo:       Mapped[str | None]     = mapped_column(String(255))
    d_locfo:       Mapped[str | None]     = mapped_column(String(255))
    d_frazfo:      Mapped[str | None]     = mapped_column(String(255))
    d_tfor:        Mapped[int | None]     = mapped_column(Integer)
    mkost:         Mapped[int | None]     = mapped_column(Integer)
    ctar1:         Mapped[str | None]     = mapped_column(String(10))
    fdesc:         Mapped[str | None]     = mapped_column(String(255))
    ccatete:       Mapped[str | None]     = mapped_column(String(20))
    tate3des:      Mapped[str | None]     = mapped_column(Text)
    # Derived during ingestion
    ateco_l1:      Mapped[str | None]     = mapped_column(String(10))
    ateco_l2:      Mapped[str | None]     = mapped_column(String(10))
    ateco_l3:      Mapped[str | None]     = mapped_column(String(10))
    # PostGIS geolocation (populated separately)
    geom:          Mapped[bytes | None]   = mapped_column(
        Geography(geometry_type="POINT", srid=4326), nullable=True
    )
    loaded_at:     Mapped[datetime]       = mapped_column(
        DateTime, server_default=func.now()
    )


# ── ATECO LOOKUP ─────────────────────────────────────────────────────────────
class AtecoLookup(Base):
    __tablename__ = "ateco_lookup"

    code:        Mapped[str]        = mapped_column(String(20), primary_key=True)
    description: Mapped[str | None] = mapped_column(Text)
    level:       Mapped[int | None] = mapped_column(SmallInteger)


# ── MEASUREMENTS — wide format Q1..Q96 ───────────────────────────────────────
# Build the Q columns dict before passing to type() / __init_subclass__
_q_columns = {
    f"q{i}": mapped_column(Float, nullable=True) for i in range(1, 97)
}


class Measurement(Base):
    __tablename__ = "measurements"

    pod:                Mapped[str]           = mapped_column(String(30), primary_key=True)
    data_misura:        Mapped[date]          = mapped_column(Date, primary_key=True)
    tipologia:          Mapped[str | None]    = mapped_column(String(10))
    matricola:          Mapped[str | None]    = mapped_column(String(50))
    tipo_rilevatore:    Mapped[str | None]    = mapped_column(String(20))
    codice_dispacciamento: Mapped[str | None] = mapped_column(String(50))
    trattamento:        Mapped[str | None]    = mapped_column(String(10))
    tensione:           Mapped[int | None]    = mapped_column(Integer)
    potenza_contrattuale: Mapped[str | None]  = mapped_column(String(20))
    k_factor:           Mapped[str | None]    = mapped_column(String(10))
    effettiva:          Mapped[bool | None]   = mapped_column(Boolean)
    validata:           Mapped[bool | None]   = mapped_column(Boolean)
    totalizzatori_presenti: Mapped[bool | None] = mapped_column(Boolean)
    intervalli_completi: Mapped[bool | None]  = mapped_column(Boolean)
    ricostruita:        Mapped[bool | None]   = mapped_column(Boolean)
    totalizzatore_f1:   Mapped[int | None]    = mapped_column(BigInteger)
    totalizzatore_f2:   Mapped[int | None]    = mapped_column(BigInteger)
    totalizzatore_f3:   Mapped[int | None]    = mapped_column(BigInteger)
    consumo_f1:         Mapped[int | None]    = mapped_column(BigInteger)
    consumo_f2:         Mapped[int | None]    = mapped_column(BigInteger)
    consumo_f3:         Mapped[int | None]    = mapped_column(BigInteger)
    source_file:        Mapped[str | None]    = mapped_column(String(255))
    loaded_at:          Mapped[datetime]      = mapped_column(
        DateTime, server_default=func.now()
    )

# Attach q1..q96 dynamically — same effect as 96 hand-written column lines
for _name, _col in _q_columns.items():
    setattr(Measurement, _name, _col)


# ── GSE REFERENCE PROFILES ───────────────────────────────────────────────────
class GseProfile(Base):
    __tablename__ = "reference_profiles_gse"

    profile_code: Mapped[str]   = mapped_column(String(10), primary_key=True)
    month_idx:    Mapped[int]   = mapped_column(SmallInteger, primary_key=True)
    hour_idx:     Mapped[int]   = mapped_column(SmallInteger, primary_key=True)
    value:        Mapped[float] = mapped_column(Float, nullable=False)


# ── ARERA REFERENCE PROFILES ─────────────────────────────────────────────────
class AreraProfile(Base):
    __tablename__ = "reference_profiles_arera"

    power_class: Mapped[str]   = mapped_column(String(40),  primary_key=True)
    market:      Mapped[str]   = mapped_column(String(100), primary_key=True)
    residenza:   Mapped[str]   = mapped_column(String(50),  primary_key=True)
    province:    Mapped[str]   = mapped_column(String(60),  primary_key=True)
    day_type:    Mapped[str]   = mapped_column(String(20),  primary_key=True)
    month_idx:   Mapped[int]   = mapped_column(SmallInteger, primary_key=True)
    hour_idx:    Mapped[int]   = mapped_column(SmallInteger, primary_key=True)
    value:       Mapped[float] = mapped_column(Float, nullable=False)
