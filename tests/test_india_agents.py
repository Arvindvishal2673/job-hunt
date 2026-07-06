import json
from unittest.mock import patch, MagicMock
import pytest
from job_hunter.agents.api_agents import AdzunaAgent
from job_hunter.models import JobListing
from job_hunter import config


def test_adzuna_agent_skips_when_credentials_missing():
    with patch("job_hunter.config.ADZUNA_APP_ID", ""), \
         patch("job_hunter.config.ADZUNA_APP_KEY", ""):
        agent = AdzunaAgent()
        results = agent.search(["python"])
        assert results == []


def test_adzuna_agent_handles_http_error():
    with patch("job_hunter.config.ADZUNA_APP_ID", "test_id"), \
         patch("job_hunter.config.ADZUNA_APP_KEY", "test_key"), \
         patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("HTTP 500 Server Error")
        agent = AdzunaAgent()
        results = agent.search(["python"])
        assert results == []


def test_adzuna_agent_parses_results_successfully():
    mock_response = {
        "results": [
            {
                "title": "<strong>Python Developer</strong>",
                "company": {"display_name": "Tech Corp"},
                "location": {"area": ["Karnataka", "Bengaluru"]},
                "description": "Develop python applications.",
                "redirect_url": "https://adzuna.com/job/123",
                "salary_min": 500000,
                "salary_max": 800000
            }
        ]
    }
    
    with patch("job_hunter.config.ADZUNA_APP_ID", "test_id"), \
         patch("job_hunter.config.ADZUNA_APP_KEY", "test_key"), \
         patch("requests.get") as mock_get:
        
        mock_json_func = MagicMock()
        mock_json_func.return_value = mock_response
        mock_get.return_value.json = mock_json_func
        mock_get.return_value.raise_for_status = MagicMock()
        
        agent = AdzunaAgent()
        results = agent.search(["python"])
        
        assert len(results) == 1
        job = results[0]
        assert job.title == "Python Developer"
        assert job.company == "Tech Corp"
        assert job.location == "Karnataka, Bengaluru"
        assert job.description == "Develop python applications."
        assert job.url == "https://adzuna.com/job/123"
        assert job.source == "Adzuna"
        assert job.salary == "INR 500,000 - 800,000"



def test_direct_ats_agent_parses_successfully():
    from job_hunter.agents.ats_agent import DirectATSAgent
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.side_effect = [
            {"jobs": [{"title": "Frontend Engineer", "content": "React Developer", "absolute_url": "https://greenhouse/groww/1", "location": {"name": "Bengaluru"}}]},
            [{"title": "Backend Developer", "description": "Python Engineer", "applyUrl": "https://lever/cred/2", "categories": {"location": "Bengaluru"}, "lists": []}],
        ]
        
        agent = DirectATSAgent()
        with patch("job_hunter.agents.ats_agent.GREENHOUSE_COMPANIES", [("groww", "Groww")]), \
             patch("job_hunter.agents.ats_agent.LEVER_COMPANIES", [("cred", "CRED")]):
            results = agent.search(["python", "react"])
            assert len(results) == 2
            assert results[0].company == "Groww"
            assert results[0].title == "Frontend Engineer"
            assert results[1].company == "CRED"
            assert results[1].title == "Backend Developer"

