# Deployment Setup Guide

This guide will walk you through setting up all the necessary accounts and secrets for the CI/CD pipeline.

## 📋 Prerequisites

Before starting, ensure you have accounts on:
- [GitHub](https://github.com) (with this repository)
- [Docker Hub](https://hub.docker.com)
- [Render](https://render.com)
- [HuggingFace](https://huggingface.co)

## 🐳 Docker Hub Setup

### 1. Create Docker Hub Account
Visit https://hub.docker.com and sign up or log in.

### 2. Create a Repository
1. Click **Create Repository**
2. Name: `mlops-lab2` (or your preferred name)
3. Visibility: Public or Private
4. Click **Create**

### 3. Generate Access Token
1. Go to **Account Settings** → **Security**
2. Click **New Access Token**
3. Description: `GitHub Actions - MLOps Lab`
4. Access permissions: **Read, Write, Delete**
5. Click **Generate**
6. **Copy the token immediately** (you won't see it again!)


## 🚀 Render Setup

### 1. Create Render Account
Visit https://render.com and sign up or log in.

### 2. Create a Web Service
1. Click **New +** → **Web Service**
2. Select **Deploy an existing image from a registry**
3. Click **Next**

### 3. Configure the Service
- **Image URL**: `docker.io/yourusername/mlops-lab2:latest` (replace with your Docker Hub username)
- **Name**: `mlops-lab2` (or your preferred name)
- **Region**: Choose closest to you
- **Instance Type**: Free tier is sufficient
- **Environment Variables**: None needed
- Click **Create Web Service**

### 4. Get Deploy Hook
1. Go to your service's **Settings**
2. Scroll to **Deploy Hook**
3. Copy the full URL: `https://api.render.com/deploy/srv-xxxxx?key=yyyyy`
4. **Extract only the key part** (everything after `key=`)
   - Example: If URL is `https://api.render.com/deploy/srv-abc123?key=xyz789`
   - Your `RENDER_DEPLOY_HOOK_KEY` is: `srv-abc123?key=xyz789`



## 🤗 HuggingFace Setup

### 1. Create HuggingFace Account
Visit https://huggingface.co and sign up or log in.

### 2. Create a Space
1. Click your profile → **New Space**
2. Space name: `mlops-lab2` (or your preferred name)
3. License: MIT
4. Space SDK: **Gradio**
5. Visibility: Public
6. Click **Create Space**

### 3. Generate Access Token
1. Go to **Settings** → **Access Tokens**
2. Click **New token**
3. Name: `GitHub Actions - MLOps Lab`
4. Role: **Write**
5. Click **Generate token**
6. **Copy the token immediately**


### 5. Initialize Space Repository
Your Space will have a Git repository at:
```
https://huggingface.co/spaces/YOUR_USERNAME/mlops-lab2
```

## 🔐 GitHub Secrets Setup

### 1. Navigate to Repository Settings
1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**

### 2. Add Repository Secrets
Click **New repository secret** for each of the following:

#### DOCKERHUB_USERNAME
- Name: `DOCKERHUB_USERNAME`
- Value: Your Docker Hub username

#### DOCKERHUB_TOKEN
- Name: `DOCKERHUB_TOKEN`
- Value: Your Docker Hub access token

#### RENDER_DEPLOY_HOOK_KEY
- Name: `RENDER_DEPLOY_HOOK_KEY`
- Value: Your Render deploy hook key (from step 4 in Render setup)

#### HF_USERNAME
- Name: `HF_USERNAME`
- Value: Your HuggingFace username

#### HF_TOKEN
- Name: `HF_TOKEN`
- Value: Your HuggingFace write access token

### 3. Verify Secrets
You should now have 5 secrets configured:
- ✅ DOCKERHUB_USERNAME
- ✅ DOCKERHUB_TOKEN
- ✅ RENDER_DEPLOY_HOOK_KEY
- ✅ HF_USERNAME
- ✅ HF_TOKEN

## 🔧 Update Configuration Files

### 1. Update Dockerfile (if needed)
If you changed the Docker Hub repository name, ensure it matches in your workflow.

### 2. Update ci.yml
Edit `.github/workflows/ci.yml`:

```yaml
# Line 55-56: Update with your Docker Hub username and repo name
tags: yourusername/mlops-lab2:latest,yourusername/mlops-lab2:${{ github.sha }}
```

### 3. Update app.py
Edit `app.py` to use your Render URL:

```python
# Line 7: Update with your Render service URL
API_URL = "https://your-service-name.onrender.com/predict"
```

### 4. Update README.md
Update the badge URLs in `README.md`:

```markdown
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-mlops--lab2-blue?logo=docker)](https://hub.docker.com/r/yourusername/mlops-lab2)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/yourusername/mlops-lab2)
```

## 🧪 Test the Pipeline

### 1. Commit Changes
```bash
git add .
git commit -m "Configure CI/CD pipeline"
git push origin main
```

### 2. Monitor GitHub Actions
1. Go to your repository's **Actions** tab
2. You should see a new workflow run starting
3. Watch the three jobs: `build`, `deploy-api`, `deploy-huggingface`

### 3. Verify Deployments

#### Docker Hub
- Visit: `https://hub.docker.com/r/yourusername/mlops-lab2/tags`
- You should see two new tags: `latest` and a commit SHA

#### Render
- Visit: `https://your-service-name.onrender.com`
- You should see the API homepage
- Check: `https://your-service-name.onrender.com/health`

#### HuggingFace
- Visit: `https://huggingface.co/spaces/yourusername/mlops-lab2`
- The Gradio interface should be live
- Try uploading an image to test

## 🐛 Troubleshooting

### Docker Hub Push Fails
- Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are correct
- Ensure the token has write permissions
- Check Docker Hub repository exists

### Render Deployment Fails
- Verify `RENDER_DEPLOY_HOOK_KEY` includes both service ID and key
- Ensure Render service is configured to pull from correct Docker image
- Check Render service logs

### HuggingFace Deployment Fails
- Verify `HF_USERNAME` and `HF_TOKEN` are correct
- Ensure token has write permissions
- Check HuggingFace Space exists
- Verify Space name matches in workflow

### Workflow Permissions Error
If you see "refusing to allow an OAuth App to create or update workflow":
1. Go to Repository **Settings** → **Actions** → **General**
2. Under **Workflow permissions**, select **Read and write permissions**
3. Check **Allow GitHub Actions to create and approve pull requests**
4. Click **Save**

## 🎉 Success Checklist

Once everything is working, you should have:
- ✅ GitHub Actions workflow passing (all 3 jobs green)
- ✅ Docker image on Docker Hub with `latest` and SHA tags
- ✅ API running on Render at your service URL
- ✅ Gradio GUI live on HuggingFace Spaces
- ✅ All badges in README working
- ✅ Test image successfully classified via GUI

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Render Documentation](https://render.com/docs)
- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs)

---

**Need Help?** Check the logs in GitHub Actions for detailed error messages, or refer to the troubleshooting section above.
