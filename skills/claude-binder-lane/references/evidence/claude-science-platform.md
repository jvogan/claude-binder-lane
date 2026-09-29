# Claude Science platform capabilities

Claude Science provides a changing catalogue of native skills, connected compute and registered model endpoints. Discover the tools and provider state in the scientist's own session before selecting a route.

The [shipped-skills snapshot](claude-science-platform-skills.md) records the 29 skills present in the 0.1.41 release inspected on 2026-08-29. An organization can add other skills; this binder package does not assume any optional extension is installed.

## Compute and model endpoints

Claude Science can use connected local or cloud compute and registered model endpoints. Provider accounts, hardware, credentials, egress policy and endpoint registrations belong to the scientist's session. The binder plan records the selected route and asks for approval before paid work or new data egress.

## Network

An NCBI sequence-retrieval route contacts an external service. Claude Science records the exact endpoint and data destination when it selects that route. No NCBI retrieval implementation ships in this package.
