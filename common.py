"""Shared helpers: Neo4j connection (password from $NEO4J_PASSWORD, never stored)."""
import os
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DATABASE = os.environ.get("NEO4J_DATABASE", "entity-resolution")
USER = os.environ.get("NEO4J_USER", "neo4j")


def driver():
    return GraphDatabase.driver(URI, auth=(USER, os.environ["NEO4J_PASSWORD"]), notifications_min_severity="OFF")


def run(drv, query, **params):
    with drv.session(database=DATABASE) as s:
        return s.run(query, **params).data()
