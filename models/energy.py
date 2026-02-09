"""
Energy data models for EcoHome Energy Advisor
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, Float, DateTime, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class EnergyUsage(Base):
    """Model for energy consumption data"""
    __tablename__ = "energy_usage"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    consumption_kwh = Column(Float, nullable=False)
    device_type = Column(String(50), nullable=True)  # e.g., "EV", "HVAC", "appliance"
    device_name = Column(String(100), nullable=True)  # e.g., "Tesla Model 3", "Main AC"
    cost_usd = Column(Float, nullable=True)  # Cost at time of usage
    
    def __repr__(self):
        return f"<EnergyUsage(timestamp={self.timestamp}, consumption={self.consumption_kwh}kWh, device={self.device_name})>"

class SolarGeneration(Base):
    """Model for solar generation data"""
    __tablename__ = "solar_generation"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    generation_kwh = Column(Float, nullable=False)
    weather_condition = Column(String(50), nullable=True)  # e.g., "sunny", "cloudy", "rainy"
    temperature_c = Column(Float, nullable=True)
    solar_irradiance = Column(Float, nullable=True)  # W/m²
    
    def __repr__(self):
        return f"<SolarGeneration(timestamp={self.timestamp}, generation={self.generation_kwh}kWh, weather={self.weather_condition})>"

class DatabaseManager:
    """Database manager for EcoHome energy data"""
    
    def __init__(self, db_path: str = "data/energy_data.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        print(f"Database tables created at {self.db_path}")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def add_usage_record(self, timestamp: datetime, consumption_kwh: float, 
                        device_type: str = None, device_name: str = None, cost_usd: float = None):
        """Add energy usage record"""
        session = self.get_session()
        try:
            record = EnergyUsage(
                timestamp=timestamp,
                consumption_kwh=consumption_kwh,
                device_type=device_type,
                device_name=device_name,
                cost_usd=cost_usd
            )
            session.add(record)
            session.commit()
            return record
        finally:
            session.close()
    
    def add_generation_record(self, timestamp: datetime, generation_kwh: float,
                            weather_condition: str = None, temperature_c: float = None,
                            solar_irradiance: float = None):
        """Add solar generation record"""
        session = self.get_session()
        try:
            record = SolarGeneration(
                timestamp=timestamp,
                generation_kwh=generation_kwh,
                weather_condition=weather_condition,
                temperature_c=temperature_c,
                solar_irradiance=solar_irradiance
            )
            session.add(record)
            session.commit()
            return record
        finally:
            session.close()
    
    def get_usage_by_date_range(self, start_date: datetime, end_date: datetime):
        """Get energy usage records within date range"""
        session = self.get_session()
        try:
            return session.query(EnergyUsage).filter(
                EnergyUsage.timestamp >= start_date,
                EnergyUsage.timestamp <= end_date
            ).order_by(EnergyUsage.timestamp).all()
        finally:
            session.close()
    
    def get_generation_by_date_range(self, start_date: datetime, end_date: datetime):
        """Get solar generation records within date range"""
        session = self.get_session()
        try:
            return session.query(SolarGeneration).filter(
                SolarGeneration.timestamp >= start_date,
                SolarGeneration.timestamp <= end_date
            ).order_by(SolarGeneration.timestamp).all()
        finally:
            session.close()
    
    def get_recent_usage(self, hours: int = 24):
        """Get recent usage records"""
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return self.get_usage_by_date_range(start_time, end_time)
    
    def get_recent_generation(self, hours: int = 24):
        """Get recent solar generation records"""
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return self.get_generation_by_date_range(start_time, end_time)
