#!/usr/bin/env python3
"""
Database initialization script for Urza service.

This script creates the necessary PostgreSQL tables for storing blueprint
and compiled kernel metadata.
"""

import os
import sys
import logging
from typing import Optional
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection(database: Optional[str] = None) -> psycopg2.extensions.connection:
    """
    Creates a connection to PostgreSQL database.
    
    Args:
        database: Database name (optional, defaults to env var or 'urza_db')
        
    Returns:
        psycopg2 connection object
        
    Raises:
        psycopg2.Error: If connection fails
    """
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    }
    
    if database:
        db_config['database'] = database
    else:
        db_config['database'] = os.getenv('POSTGRES_DB', 'urza_db')
    
    logger.info("Connecting to PostgreSQL at %s:%s", db_config['host'], db_config['port'])
    return psycopg2.connect(**db_config)


def create_database_if_not_exists(database_name: str) -> None:
    """
    Creates the database if it doesn't exist.
    
    Args:
        database_name: Name of the database to create
    """
    try:
        # Connect to postgres database to create target database
        conn = get_db_connection('postgres')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (database_name,)
        )
        
        if cursor.fetchone():
            logger.info("Database '%s' already exists", database_name)
        else:
            cursor.execute(f'CREATE DATABASE "{database_name}"')
            logger.info("Database '%s' created successfully", database_name)
        
        cursor.close()
        conn.close()
    except psycopg2.Error as e:
        logger.error("Error creating database '%s': %s", database_name, e)
        raise


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """
    Creates the required tables for Urza service.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    try:
        # Create blueprints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blueprints (
                id VARCHAR(64) PRIMARY KEY,
                status VARCHAR(32) NOT NULL,
                architecture_ir_ref TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
            );
        """)
        
        # Create index on blueprint status for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_blueprints_status 
            ON blueprints(status);
        """)
        
        # Create compiled_kernels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compiled_kernels (
                id VARCHAR(128) PRIMARY KEY,
                blueprint_id VARCHAR(64) NOT NULL REFERENCES blueprints(id) ON DELETE CASCADE,
                status VARCHAR(32) NOT NULL,
                compilation_pipeline VARCHAR(32) NOT NULL,
                kernel_binary_ref TEXT NOT NULL,
                validation_report JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
            );
        """)
        
        # Create indexes on compiled_kernels for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_compiled_kernels_blueprint_id 
            ON compiled_kernels(blueprint_id);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_compiled_kernels_status 
            ON compiled_kernels(status);
        """)
        
        # Create trigger to update updated_at column
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = timezone('utc', now());
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        # Create triggers for both tables
        cursor.execute("""
            DROP TRIGGER IF EXISTS update_blueprints_updated_at ON blueprints;
            CREATE TRIGGER update_blueprints_updated_at
                BEFORE UPDATE ON blueprints
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        cursor.execute("""
            DROP TRIGGER IF EXISTS update_compiled_kernels_updated_at ON compiled_kernels;
            CREATE TRIGGER update_compiled_kernels_updated_at
                BEFORE UPDATE ON compiled_kernels
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        conn.commit()
        logger.info("Database tables created successfully")
        
    except psycopg2.Error as e:
        logger.error("Error creating tables: %s", e)
        conn.rollback()
        raise
    finally:
        cursor.close()


def init_database() -> None:
    """
    Initializes the Urza database with required tables and indexes.
    """
    database_name = os.getenv('POSTGRES_DB', 'urza_db')
    
    try:
        # Create database if it doesn't exist
        create_database_if_not_exists(database_name)
        
        # Connect to the target database and create tables
        conn = get_db_connection(database_name)
        create_tables(conn)
        conn.close()
        
        logger.info("Urza database initialized successfully")
        
    except psycopg2.Error as e:
        logger.error("Failed to initialize database: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    init_database()
