# Notebook: 05-dotnet-agent-framework

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/05-agentic-rag/code_samples/05-dotnet-agent-framework.ipynb

---

# 🔍 Enterprise RAG with Microsoft Foundry (.NET)

## 📋 Learning Objectives

This notebook demonstrates how to build enterprise-grade Retrieval-Augmented Generation (RAG) systems using the Microsoft Agent Framework in .NET with Microsoft Foundry. You'll learn to create production-ready agents that can search through documents and provide accurate, context-aware responses with enterprise security and scalability.

**Enterprise RAG Capabilities You'll Build:**
- 📚 **Document Intelligence**: Advanced document processing with Azure AI services
- 🔍 **Semantic Search**: High-performance vector search with enterprise features
- 🛡️ **Security Integration**: Role-based access and data protection patterns
- 🏢 **Scalable Architecture**: Production-ready RAG systems with monitoring

## 🎯 Enterprise RAG Architecture

### Core Enterprise Components
- **Microsoft Foundry**: Managed enterprise AI platform with security and compliance
- **Persistent Agents**: Stateful agents with conversation history and context management
- **Vector Store Management**: Enterprise-grade document indexing and retrieval
- **Identity Integration**: Azure AD authentication and role-based access control

### .NET Enterprise Benefits
- **Type Safety**: Compile-time validation for RAG operations and data structures
- **Async Performance**: Non-blocking document processing and search operations
- **Memory Management**: Efficient resource utilization for large document collections
- **Integration Patterns**: Native Azure service integration with dependency injection

## 🏗️ Technical Architecture

### Enterprise RAG Pipeline
```csharp
Document Upload → Security Validation → Vector Processing → Index Creation
                      ↓                    ↓                  ↓
User Query → Authentication → Semantic Search → Context Ranking → AI Response
```

### Core .NET Components
- **Azure.AI.Agents.Persistent**: Enterprise agent management with state persistence
- **Azure.Identity**: Integrated authentication for secure Azure service access
- **Microsoft.Agents.AI.AzureAI**: Azure-optimized agent framework implementation
- **System.Linq.Async**: High-performance asynchronous LINQ operations

## 🔧 Enterprise Features & Benefits

### Security & Compliance
- **Azure AD Integration**: Enterprise identity management and authentication
- **Role-Based Access**: Fine-grained permissions for document access and operations
- **Data Protection**: Encryption at rest and in transit for sensitive documents
- **Audit Logging**: Comprehensive activity tracking for compliance requirements

### Performance & Scalability
- **Connection Pooling**: Efficient Azure service connection management
- **Async Processing**: Non-blocking operations for high-throughput scenarios
- **Caching Strategies**: Intelligent caching for frequently accessed documents
- **Load Balancing**: Distributed processing for large-scale deployments

### Management & Monitoring
- **Health Checks**: Built-in monitoring for RAG system components
- **Performance Metrics**: Detailed analytics on search quality and response times
- **Error Handling**: Comprehensive exception management with retry policies
- **Configuration Management**: Environment-specific settings with validation

## ⚙️ Prerequisites & Setup

**Development Environment:**
- .NET 9.0 SDK or higher
- Visual Studio 2022 or VS Code with C# extension
- Azure subscription with AI Foundry access

**Required NuGet Packages:**
```xml
<PackageReference Include="Microsoft.Extensions.AI" Version="9.9.0" />
<PackageReference Include="Azure.AI.Agents.Persistent" Version="1.2.0-beta.5" />
<PackageReference Include="Azure.Identity" Version="1.15.0" />
<PackageReference Include="System.Linq.Async" Version="6.0.3" />
<PackageReference Include="DotNetEnv" Version="3.1.1" />
```

**Azure Authentication Setup:**
```bash
# Install Azure CLI and authenticate
az login
az account set --subscription "your-subscription-id"
```

**Environment Configuration (.env file):**
```env
# Microsoft Foundry configuration (automatically handled via Azure CLI)
# Ensure you're authenticated to the correct Azure subscription
```

## 📊 Enterprise RAG Patterns

### Document Management Patterns
- **Bulk Upload**: Efficient processing of large document collections
- **Incremental Updates**: Real-time document addition and modification
- **Version Control**: Document versioning and change tracking
- **Metadata Management**: Rich document attributes and taxonomy

### Search & Retrieval Patterns
- **Hybrid Search**: Combining semantic and keyword search for optimal results
- **Faceted Search**: Multi-dimensional filtering and categorization
- **Relevance Tuning**: Custom scoring algorithms for domain-specific needs
- **Result Ranking**: Advanced ranking with business logic integration

### Security Patterns
- **Document-Level Security**: Fine-grained access control per document
- **Data Classification**: Automatic sensitivity labeling and protection
- **Audit Trails**: Comprehensive logging of all RAG operations
- **Privacy Protection**: PII detection and redaction capabilities

## 🔒 Enterprise Security Features

### Authentication & Authorization
```csharp
// Azure AD integrated authentication
var credential = new AzureCliCredential();
var agentsClient = new PersistentAgentsClient(endpoint, credential);

// Role-based access validation
if (!await ValidateUserPermissions(user, documentId))
{
    throw new UnauthorizedAccessException("Insufficient permissions");
}
```

### Data Protection
- **Encryption**: End-to-end encryption for documents and search indices
- **Access Controls**: Integration with Azure AD for user and group permissions
- **Data Residency**: Geographic data location controls for compliance
- **Backup & Recovery**: Automated backup and disaster recovery capabilities

## 📈 Performance Optimization

### Async Processing Patterns
```csharp
// Efficient async document processing
await foreach (var document in documentStream.AsAsyncEnumerable())
{
    await ProcessDocumentAsync(document, cancellationToken);
}
```

### Memory Management
- **Streaming Processing**: Handle large documents without memory issues
- **Resource Pooling**: Efficient reuse of expensive resources
- **Garbage Collection**: Optimized memory allocation patterns
- **Connection Management**: Proper Azure service connection lifecycle

### Caching Strategies
- **Query Caching**: Cache frequently executed searches
- **Document Caching**: In-memory caching for hot documents
- **Index Caching**: Optimized vector index caching
- **Result Caching**: Intelligent caching of generated responses

## 📊 Enterprise Use Cases

### Knowledge Management
- **Corporate Wiki**: Intelligent search across company knowledge bases
- **Policy & Procedures**: Automated compliance and procedure guidance
- **Training Materials**: Intelligent learning and development assistance
- **Research Databases**: Academic and research paper analysis systems

### Customer Support
- **Support Knowledge Base**: Automated customer service responses
- **Product Documentation**: Intelligent product information retrieval
- **Troubleshooting Guides**: Contextual problem-solving assistance
- **FAQ Systems**: Dynamic FAQ generation from document collections

### Regulatory Compliance
- **Legal Document Analysis**: Contract and legal document intelligence
- **Compliance Monitoring**: Automated regulatory compliance checking
- **Risk Assessment**: Document-based risk analysis and reporting
- **Audit Support**: Intelligent document discovery for audits

## 🚀 Production Deployment

### Monitoring & Observability
- **Application Insights**: Detailed telemetry and performance monitoring
- **Custom Metrics**: Business-specific KPI tracking and alerting
- **Distributed Tracing**: End-to-end request tracking across services
- **Health Dashboards**: Real-time system health and performance visualization

### Scalability & Reliability
- **Auto-Scaling**: Automatic scaling based on load and performance metrics
- **High Availability**: Multi-region deployment with failover capabilities
- **Load Testing**: Performance validation under enterprise load conditions
- **Disaster Recovery**: Automated backup and recovery procedures

Ready to build enterprise-grade RAG systems that can handle sensitive documents at scale? Let's architect intelligent knowledge systems for the enterprise! 🏢📖✨

```python
#r "nuget: Microsoft.Extensions.AI, 9.9.1"
```

```python
#r "nuget: Azure.AI.Agents.Persistent, 1.2.0-beta.5"
#r "nuget: Azure.Identity, 1.15.0"
#r "nuget: System.Linq.Async, 6.0.3"
```

```python
#r "nuget: Microsoft.Agents.AI.AzureAI, 1.0.0-preview.251001.3"
```

```python
#r "nuget: Microsoft.Agents.AI, 1.0.0-preview.251001.3"
```

```python
#r "nuget: DotNetEnv, 3.1.1"
```

```python
using System;
using System.Linq;
using Azure.AI.Agents.Persistent;
using Azure.Identity;
using Microsoft.Agents.AI;
```

```python
 using DotNetEnv;
```

```python
Env.Load("../../../.env");
```

```python
var azure_foundry_endpoint = Environment.GetEnvironmentVariable("AZURE_AI_PROJECT_ENDPOINT") ?? throw new InvalidOperationException("AZURE_AI_PROJECT_ENDPOINT is not set.");
var azure_foundry_model_id = Environment.GetEnvironmentVariable("AZURE_AI_MODEL_DEPLOYMENT_NAME") ?? "gpt-4.1-mini";
```

```python
string pdfPath = "./document.md";
```

```python
using System.IO;

async Task<Stream> OpenImageStreamAsync(string path)
{
	return await Task.Run(() => File.OpenRead(path));
}

var pdfStream = await OpenImageStreamAsync(pdfPath);
```

```python
var persistentAgentsClient = new PersistentAgentsClient(azure_foundry_endpoint, new AzureCliCredential());
```

```python
PersistentAgentFileInfo fileInfo = await persistentAgentsClient.Files.UploadFileAsync(pdfStream, PersistentAgentFilePurpose.Agents, "demo.md");
```

```python
PersistentAgentsVectorStore fileStore =
            await persistentAgentsClient.VectorStores.CreateVectorStoreAsync(
                [fileInfo.Id],
                metadata: new Dictionary<string, string>() { { "agentkey", bool.TrueString } });
```

```python
PersistentAgent agentModel = await persistentAgentsClient.Administration.CreateAgentAsync(
            azure_foundry_model_id,
            name: "DotNetRAGAgent",
            tools: [new FileSearchToolDefinition()],
            instructions: """
                You are an AI assistant designed to answer user questions using only the information retrieved from the provided document(s).

                - If a user's question cannot be answered using the retrieved context, **you must clearly respond**: 
                "I'm sorry, but the uploaded document does not contain the necessary information to answer that question."
                - Do not answer from general knowledge or reasoning. Do not make assumptions or generate hypothetical explanations.
                - Do not provide definitions, tutorials, or commentary that is not explicitly grounded in the content of the uploaded file(s).
                - If a user asks a question like "What is a Neural Network?", and this is not discussed in the uploaded document, respond as instructed above.
                - For questions that do have relevant content in the document (e.g., Contoso's travel insurance coverage), respond accurately, and cite the document explicitly.

                You must behave as if you have no external knowledge beyond what is retrieved from the uploaded document.
                """,
            toolResources: new()
            {
                FileSearch = new()
                {
                    VectorStoreIds = { fileStore.Id },
                }
            },
            metadata: new Dictionary<string, string>() { { "agentkey", bool.TrueString } });
```

```python
AIAgent agent = await persistentAgentsClient.GetAIAgentAsync(agentModel.Id);
```

```python
AgentThread thread = agent.GetNewThread();
```

```python
Console.WriteLine(await agent.RunAsync("Can you explain Contoso's travel insurance coverage?", thread));
```