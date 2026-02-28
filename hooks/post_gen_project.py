"""
Post-generation hook for cookiecutter.
Runs after the project is generated to clean up cloud- and CI/CD-specific folders.

What it does:
  1. Keeps the correct .github/workflows/ CI/CD file and deletes the other
  2. Deletes .azure/pipelines/ if the LOB is on AWS (Azure DevOps is Azure-only)
  3. Deletes .azure/pipelines/ if cloud_provider=azure but cicd_platform=github_actions
"""

import os
import shutil

CLOUD_PROVIDER = "{{cookiecutter.cloud_provider}}"
CICD_PLATFORM = "{{cookiecutter.cicd_platform}}"

project_dir = os.getcwd()


def remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


workflows_dir = os.path.join(project_dir, ".github", "workflows")
azure_workflow = os.path.join(workflows_dir, "azure", "bundle-deploy.yml")
aws_workflow = os.path.join(workflows_dir, "aws", "bundle-deploy.yml")
azure_pipelines_dir = os.path.join(project_dir, ".azure")

if CLOUD_PROVIDER == "azure":
    # Move the azure workflow to .github/workflows/ and delete both subfolders
    move(azure_workflow, os.path.join(workflows_dir, "bundle-deploy.yml"))
    remove(os.path.join(workflows_dir, "azure"))
    remove(os.path.join(workflows_dir, "aws"))

    # Keep or remove .azure/pipelines/ based on chosen CI/CD platform
    if CICD_PLATFORM == "github_actions":
        remove(azure_pipelines_dir)

elif CLOUD_PROVIDER == "aws":
    # Move the aws workflow to .github/workflows/ and delete both subfolders
    move(aws_workflow, os.path.join(workflows_dir, "bundle-deploy.yml"))
    remove(os.path.join(workflows_dir, "azure"))
    remove(os.path.join(workflows_dir, "aws"))

    # AWS LOBs always use GitHub Actions — Azure DevOps not applicable
    remove(azure_pipelines_dir)
