# Human head anatomy models for facial biomechanics

## Assessment

**There are better sources for individual parts of the current anatomy pipeline, but no verified, obtainable package in this assessment supplies all of the following together: a complete human head, specimen-derived facial connective tissues, explicit muscle-to-dermis attachment topology, spatially resolved muscle fascicles, and a simulation-ready discretization.** The useful choices differ substantially in what they improve.

The most promising route to measured muscle architecture is the University of Toronto's digitized facial-muscle work. The most relevant recent lead for retaining ligaments is Park and Noël's reconstruction from real-color sections. Both warrant a data-access enquiry; publication of the research does not establish availability of its underlying models. ArtiSynth is the strongest immediately inspectable simulation baseline. Atomedge Tommy is a promising commercial anatomy scaffold, subject to examination of an actual facial sample. MIDA supplies organized image-based labels, but its standard license substantially limits redistribution and face-image publication.[^1][^2][^3][^4][^5][^6]

The recommended decision is to **retain the current model as an experimental baseline while evaluating replacements against specific missing fields**. A more detailed commercial surface model alone would leave the largest scientific uncertainties unresolved. Conversely, a smaller regional dataset with measured fascicles and insertions could improve the relevant force paths more than a visually complete head.

This assessment covers research and commercial sources available as of 12 September 2026. Prices are public asking prices where available; enquiry-based products are not assigned estimated prices. “Not verified” means the public documentation or inspected files did not establish a feature or data grant, rather than proof that the supplier has no such data. No purchase, supplier contact, or model migration is included.

## Current Melon and Apple baseline

The baseline below distinguishes stored arrays from fields actually used by the solver. Melon was inspected at `3eccfe3578420d9443fb541f9feb797ab5939307`; Apple at `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`, including the September experiments available at the time.

| Component | Current construction | Consequence for a replacement |
| --- | --- | --- |
| Original anatomy | A GLB containing named skin, skeletal, and muscle surfaces; 293 muscle components after extraction and repair | Better provenance and clean semantic objects would already be useful. Polygon count alone is insufficient. |
| Skin and bones | XYZ skin topology and Sculptor cranium/mandible are separately registered to the GLB anatomy using Wrap | An internally registered anatomy package could eliminate these particular cross-source registrations. Subject personalization and meshing would remain. |
| SMAS | Constructed from the skin and selected muscle surfaces, with a maximum 18 mm skin-normal span and interpolated offsets | This is an envelope generated from geometry, not a segmented anatomical structure. |
| Aponeurosis | Samples inside that envelope but outside source muscles become `AponeurosisFraction` | The name denotes a generated material partition. It does not establish a real aponeurosis, continuity, or orientation. |
| Muscle labels | Muscle surfaces are sampled into a background tetrahedral mesh | Useful semantics, but internal muscle boundaries are not conforming mesh interfaces. |
| Skin coupling | Skin triangles reuse volume nodes through `GlobalPointId` | Forces can reach skin through bulk tissue. There is no separate insertion map, retaining-ligament representation, or sliding law. |
| Fibers | Melon's transient inertia frames do not survive into its final mesh; Apple estimates `ActivationFiber` using weighted PCA and ellipse tangents | Neither constitutes measured fascicle anatomy. |
| Expression targets | Faceform expression displacements are registered and transferred to the model | Targets are not measured motion from the anatomy donor. |

The final Melon volume has 599,998 points and 3,190,515 tetrahedra. Apple's face subset has 228,660 points and 1,146,517 tetrahedra; its current fixture activates 120,020 cells in a whitelist of 35 facial-expression components. These are substantial, usable computational assets. Their principal deficit is the anatomical evidence behind internal force paths, rather than a lack of mesh density.

Two implementation details matter when judging an upgrade. `AponeurosisFraction` currently weights an isotropic passive material. `ActivationFiber` is loaded, but only fiber-constrained activation modes use its values; `Raw6`, `G6`, and `G5` do not. A new fiber field or attachment file therefore needs an explicit constitutive or interface consumer before it can improve the simulation.

The GLB's likely origin is 3D4SCI's *Complete Human Head Anatomy* on Sketchfab: its scene root and triangle count closely match that listing. This is an identification inference, since the repository does not preserve the acquisition URL, author, version, or license. Recovering that record would improve the baseline's traceability independently of any replacement.[^7]

## Anatomical requirements

### Connective tissue is not one interchangeable layer

An acquisition specification should distinguish the scalp's galea/epicranial aponeurosis, named deep fasciae, intramuscular aponeuroses and tendons, facial retaining ligaments, subcutaneous fibrous septa or retinacula cutis, and direct or interdigitating dermal/mucosal muscle insertions. An atlas that includes the galea has not thereby supplied the connective anatomy needed around the nasolabial fold or mouth corner.

SMAS terminology itself is contested. Ghassemi and colleagues described a continuous but regionally varying three-dimensional fibrous network connecting muscles and dermis. Sandulescu and colleagues reconstructed different septal architectures in midfacial tissue blocks. A larger later investigation by Minelli and colleagues, using dissection, histology, and plastination, disputed a distinct continuous aponeurotic sheet between the flat mimetic muscles.[^8][^9][^10] These findings support a practical requirement: **request named regional structures, their observed continuities, and their reconstruction method; do not treat a single continuous aponeurotic sheet as universal anatomical ground truth.** This does not imply that facial connective tissues or their force transmission are absent, or that one study settles the terminology.

### Attachment and fiber data must be computationally identifiable

A useful attachment is a region or relation tied to both the muscle/bundle and its receiving tissue. Its location, extent, continuity, and reference configuration must be recoverable. A textual origin/insertion description, proximity of two meshes, a rendering blend mask, or a skinning weight can help an artist or annotator but does not establish that mechanical relation.

Likewise, surface striations and textures are not a volumetric muscle fiber field. A meaningful upgrade supplies digitized fascicle centerlines or directions with a stated measurement method. A single long axis, an artist-drawn path, an ellipsoidal tangent field, or a direction fitted to motion remains a modeled quantity. Each may be useful, but they answer different research questions.

## Candidate comparison

“Modeled” denotes a computational construction; “measured” denotes a specimen/imaging-based observation with the stated limitations. The table separates publicly available assets from research results whose coordinates have not been obtained.

| Candidate | Usable asset/access | Facial muscles and common geometry | Connective tissue / dermal attachments | Fiber architecture | Best role |
| --- | --- | --- | --- | --- | --- |
| **ArtiSynth face models** | Public source, volume meshes, muscle node paths; no purchase fee | Explicit face models with skull/jaw support | Modeled fixation and selected attachment node sets; no complete measured dermal map | Explicit modeled paths and derived element directions | Immediate simulation comparison |
| **Toronto Li/Agur model** | Published model; underlying coordinate license/access unverified | Digitized expression and masticatory muscles registered to skeletal data | Musculotendinous/aponeurotic architecture; skin removed during dissection | Specimen-digitized bundles | Highest-priority fascicle data enquiry |
| **Park–Noël retaining ligaments** | Open-access article; separate model files not found | Regional head reconstruction from real-color sections | Reconstructed named retaining ligaments; no complete muscle-to-dermis map established | Ligament orientation discussed; no whole-face muscle field | Highest-priority ligament data enquiry |
| **Atomedge Tommy** | Commercial, quote-based; polygon formats | Detailed named facial structures; integrated systems claimed | Explicit fascia object inventory; measurement of each fascia and dermal topology unverified | Numerical field not documented | Commercial scaffold sample |
| **Zygote** | Commercial licensed polygon/CAD products | Extensive anatomy; product-specific facial coverage must be checked | Some named connective anatomy; full insertion topology unverified | Numerical facial field not documented | Alternative commercial sample |
| **MIDA v1.0** | Free except handling fees; request form; restricted license | Aligned labeled voxels and surfaces; manual/atlas-assisted facial segmentation | Galea label; generated skin thickness; no complete dermal map | Raw MR/DT data excluded; no facial field verified | Internal geometry/label reference |
| **Z-Anatomy** | Public Blender/FBX assets; CC BY-SA 4.0 | Facial-expression geometry verified in distributed FBX | Epicranial aponeurosis and selected fascia geometry; no validated attachment map | No measured field established | Open semantic surface reference |
| **BodyParts3D 3.0 / OMFAtlas** | Public surfaces; version-specific licenses | Older BodyParts3D includes expression muscles; OMFAtlas registers them to a newer base | No verified detailed insertion/septal architecture | No measured field established | Open coverage and labeling reference |
| **eFace / eFTD-VP** | Published FE template; reusable data access unverified | Skin/mucosa and facial muscles, template registration | Useful layer/template construction; required attachment details unverified | Follow-up explicitly lacks detailed fiber orientation | Academic FE-template enquiry |
| **Auckland Wu–Hung–Mithraratne** | Published layered facial FE models; download not verified | Explicit facial muscles and soft-tissue layers | Modeled SMAS, muscle/fat arrangements and interfaces | Directions modeled; specimen measurements not established here | Academic model enquiry |
| **Visible Human / Visible Korean / Chinese Visible Human** | Raw section/imaging resources; access varies | Real anatomical observations, varying prepared labels | Potential to reconstruct relevant structures | Would require extraction/measurement | Original anatomy project |

Sources for each row and its limits are detailed below. No ranking combines polygon detail, scientific validity, and import effort into a single numerical score.

## Open and institutional candidates

### ArtiSynth: the strongest immediate simulation baseline

The official models package is downloadable with source; the page currently lists release 3.9 and also directs users to the development repository. Its version must match ArtiSynth core. The models package has its own redistribution and citation conditions, so the core software license should not be substituted for the data/model license.[^3][^11]

The distributed face directory contains actual node/element meshes, muscle-path definitions, and named attachment files. `BadinFaceDemo` loads the face and muscle topology, attaches it to skull and jaw, and fixes selected node sets including a zygomatic-ligament set. `BadinFemMuscleFaceDemo` derives element directions from muscle paths and identifies surrounding muscle elements. These are executable modeling choices, not merely anatomy illustrations.[^12][^13]

This directly helps with a missing part of the current workflow: an inspectable connection between anatomy labels, muscle paths, finite elements, and boundary conditions. It could support an independent reference implementation or conversion experiment. It does **not** prove measured whole-face fascicles or ligament geometry: a fixed node set is not a segmented ligament, and paths converted into local directions are not a specimen-derived field. Public integrated face–tongue–jaw and FRANK demonstrations also emphasize speech, mastication, and related orofacial function; their coverage should not be assumed identical to the current expression-muscle set.[^14]

**Recommendation:** use a specific public face model for a small import and actuation comparison. Preserve its original topology and attribution, enumerate its muscle coverage, and document each modeled constraint. No ArtiSynth simulation was executed for this assessment, so runtime compatibility and output quality remain to be tested.

### Toronto: the most relevant measured fascicle architecture

Li, Tran, Bibliowicz, Khan, Mogk, and Agur describe digitization of 22 named facial-expression and masticatory muscles. Bundles were traced through dissection using a MicroScribe, reconstructed as curves, and registered to CT-derived skeletal geometry. The work preserves asymmetry, interdigitation, and musculotendinous/aponeurotic relationships instead of imposing homogeneous directions.[^1]

The full chapter was retrievable from the publisher during this assessment. That is access to the publication, **not a grant to download or reuse the underlying coordinates**. No public geometry repository or supplement containing the bundle data was verified. The corresponding University of Toronto laboratory is the appropriate data-access route.[^15]

The skin was removed before digitization. Therefore, even successful acquisition should not be described as a complete intact skin–muscle attachment atlas. The specific missing deliverable is a registration and anatomical relation between bundle termini, connective tissue, and dermal/mucosal regions. Request original point trajectories and reference frames, not only rendered tubes or screenshots.[^1]

**Recommendation:** give this enquiry the highest priority if measured activation directions and real intramuscular connective architecture are the main objective. Acquisition remains a collaboration/licensing question; no price or permission was established.

### Park and Noël: unusually relevant retaining-ligament reconstruction

The 2025 study reconstructs facial retaining ligaments from real-color sectioned images. Its reported structures include orbital, zygomatic, maxillary, platysma-auricular, masseteric, mental, mandibular, and cervicomental regions. It addresses spatial relationships and orientation relevant to skin support.[^2]

The article is open access under CC BY 4.0, as confirmed by publisher metadata. No separate image stack, annotation volume, mesh archive, or reusable 3D model download was found in the checked publisher links and metadata. The existence of 3D figures in the paper does not establish availability of those files.

**Recommendation:** enquire about the original section stack, segmented structures, common coordinate frame, and export/reuse license together. This is a strong complement to muscle-bundle data, but not evidence of a turnkey combined head. Registration to another donor would remain necessary if the two datasets were combined.

### MIDA: aligned labels, with important construction and license limits

MIDA v1.0 distributes labeled voxels in MAT/RAW/NIfTI and surfaces in STL, sampled at 500 μm. The current distribution page lists 115 structures; the paper discusses 153 anatomical structures under its counting scheme. Its facial labels include several major expression muscles. Raw MR and diffusion-tensor images are explicitly excluded from distribution.[^5][^16]

The acquisition paper's diffusion imaging concerns brain tissue; it is not evidence of a delivered facial-muscle direction field. The supplementary segmentation methods also show that epidermis and dermis were combined into a constructed constant 1.5 mm layer. Facial muscles required extensive manual discrimination using MRI and atlases. Thus the package offers coherent labels, but does not eliminate all geometric assumptions or resolve dermal insertions.[^16][^17]

The current standard license permits use but prohibits redistribution of original **and derived** model data (§2.3.2). Images may be published only with the face disguised so the individual is unrecognizable (§2.3.3). This is a material obstacle for an openly distributed face benchmark and recognizable facial renderings. The distribution page says free except handling fees; the fee amount is not published.[^5][^6]

**Recommendation:** consider MIDA for internal segmentation and geometry comparison. Resolve the intended derivative-data and face-figure uses with IT'IS before adopting it as the basis of a public research artifact. “Free” does not make this an open-data alternative.

### Z-Anatomy: a real open surface candidate

The public project distributes editable anatomy assets and identifies CC BY-SA 4.0 licensing. Inspection of the actual `MuscularSystem100.fbx`, rather than only its labels in a viewer, found Geometry objects for major expression muscles, including orbicularis oris/oculi, zygomatici, lip elevators/depressors, buccinator, risorius, mentalis, frontalis, and corrugator. Geometry objects also include epicranial aponeurosis, masseteric fascia, and temporal fascia.[^18]

This is a meaningful improvement over an atlas package that omits the expression muscles. It could provide an auditable semantic surface baseline. However, the inspected asset does not establish measured dermal insertion maps, fascicle coordinates, or a validated regional SMAS network. Mesh watertightness, interpenetrations, thickness, and biological provenance have not been qualified for FEM use. Share-alike obligations for adapted assets also need to be preserved.

**Recommendation:** the best low-cost surface inventory to inspect before buying another general anatomy mesh. Its role is geometry and labels; a full anatomical force-path reconstruction would still be a separate task.

### BodyParts3D and OMFAtlas: version matters

BodyParts3D 3.0 contains facial-expression muscles in its official parts list. Its archive identifies CC BY-SA 2.1 Japan licensing and explicitly warns about geometric incompleteness, gaps, and overlap. Findings about missing muscles in a particular BodyParts3D 4.0-based viewer should therefore not be generalized to every version.[^19]

OMFAtlas documents recovering 47 facial/masticatory muscle meshes from 3.0 and registering them to a 4.0 base using shared skeletal structures. This is useful engineering and transparent provenance, but is itself a cross-version registration, rather than a same-specimen scan. The facial subset retains its source license; the newer base's license should not be applied indiscriminately to it.[^20]

**Recommendation:** useful for a free semantic reference or checking missing muscle coverage. It offers no demonstrated shortcut to measured attachments or fibers, and its registration work resembles part of the current pipeline.

### Other published facial FE models

The **eFace** template work reconstructs skin, mucosa, and eleven facial muscles from a Chinese Visible Human source and transfers the template to other heads. It is relevant to semantic FE meshing and registration. However, the later eFTD-VP work explicitly notes that detailed fiber orientation is not represented because active muscle contraction is outside that model's scope. A reusable public data package was not verified.[^21][^22]

The **Wu–Hung–Mithraratne/Auckland** model includes skin, subcutaneous tissue, SMAS, and twenty bilateral facial muscle pairs, with muscle/fat heterogeneity and interfaces. It is closer to the desired mechanical organization than a display atlas. Its construction involves imaging, manual modeling, and assumptions; the checked sources do not establish a public distribution of measured whole-face fascicles or dermal attachment labels. Request the actual model and its construction record if pursuing this route.[^23][^24]

Both are second-tier academic enquiries. They could save substantial model-building effort if the data are available, but a paper's successful simulation does not demonstrate the anatomical fields sought here.

## Commercial candidates

Commercial products are evaluated below as potential sources of geometry and metadata. None is accepted as measured facial attachment/fiber ground truth on the strength of a render or product description.

| Product | Public price | Delivery and potential advantage | Decisive limitation |
| --- | --- | --- | --- |
| **Atomedge Tommy** | Quote required | Named polygon assembly; editable OBJ/FBX/Blender and other DCC formats; substantial facial fascia inventory | Exact current head subset, per-structure measurement provenance, dermal topology, and numerical fibers require confirmation. |
| **Zygote Solid 3D Human Head** | **USD 6,825** | STEP, IGES, Parasolid, SolidWorks, Creo; head skin, skull, muscles and connective-tissue systems | Solid CAD does not establish measured thin fascia or anatomical attachment relations. |
| **CGTrader Facial Anatomy Layers** | **USD 549** | FBX/glTF; explicitly advertises skin, superficial fat, SMAS, loose connective tissue, deep fascia/periosteum, and bone | No published specimen validation or numerical attachment/fiber data. |
| **TurboSquid head with facial fat layers, 2450228** | **USD 169** | Polygon anatomy with regional facial fat compartments | A possible supplemental reference; complete skin/skull scope and required mechanics are not established. |
| **TurboSquid Full Human Head Anatomy, 1925680** | **USD 99** | General anatomy polygon model | No demonstrated upgrade to the required evidence or mechanical fields; not established as the current GLB's source. |

Prices and formats are from the specific listings, not estimates. The CGTrader listing uses its Royalty Free License; the TurboSquid listings use its Standard License.[^4][^37][^41][^42][^43] Taxes, negotiated terms, and separate redistribution rights are not included.

### Atomedge Tommy: first commercial sample to evaluate

Atomedge documents a mixture of cadaver surface scans, MRI, dissection, specimen photography, literature, and expert input, with University of Cape Town ethics approval. This is stronger provenance than an unattributed marketplace render, although it does not prove that every structure was measured from the same donor.[^35]

The linked anatomical inventory includes named facial muscles and corresponding fascia objects, including buccinator, orbicularis oculi, zygomatici, and lip elevators, plus galea and parotid–masseteric fascia. A material version discrepancy must be resolved: the product page advertises **v3.0**, while the downloadable inventory filename identifies **v2.0**.[^4][^36]

Origin/insertion blending masks advertised for bone and connective tissue do not establish mechanical dermal attachment regions. Neither a spatial muscle fiber field nor a volume mesh is documented. The sample should show what the fascia objects and masks actually encode.[^4]

**Recommendation:** request the offered evaluation sample with a current head-only manifest. Atomedge's license categories distinguish research, commercial, and publishing uses; a written quote should cover the intended derivative FEM outputs, rather than only rendered imagery. Price is enquiry-based.[^35][^40]

### Zygote: strongest commercial CAD route

The solid head product offers editable CAD bodies, including skin, skull, muscular and connective-tissue systems. Its listed price is USD 6,825. The skin and skeleton are described as scan-derived; the related solid-muscle collection describes hand-constructed muscles fitted using atlases and CT/MRI templates.[^37][^38]

The separate polygon muscular-system inventory includes major facial-expression muscles and claims appropriate origins and insertions. That supports the product family's anatomical scope, but it does not guarantee that every object or field is delivered in the specific solid-head purchase. The exact head manifest and assembly are needed.[^39]

**Recommendation:** evaluate alongside Atomedge if reducing repair and volume-construction work is the immediate priority. CAD delivery is promising for meshing, but no tetrahedralization success or anatomical interface quality was tested here. Product-specific license terms govern the purchased files; obtain explicit terms for converted/derived meshes. There is no verified public numerical facial fiber field or measured dermal insertion map.[^37][^44]

### Marketplace layers and fat compartments

The CGTrader *Facial Anatomy Layers* listing is a concrete purchasable claim of a separate SMAS layer. Its technical verification concerns file and visual properties; it does not establish anatomical validation. A smooth named sheet could reproduce the current model's core uncertainty under a different provenance. It is therefore a geometry reference, not a recommended substitute for observed regional connective tissue.[^41]

The TurboSquid facial-fat model may help identify or visualize compartment boundaries, but its listing does not establish measured partitions or attachment semantics. The generic USD 99 head similarly lacks evidence for the missing fields. These products are lower priority than an anatomically informative sample from a specialist supplier.[^42][^43]

## Primary anatomy resources and regional studies

### Raw anatomical image collections

The **NLM Visible Human Project** provides public-domain cryosection and medical-imaging data, including additional head images. It is an accessible source for original reconstruction, but raw slices do not supply ready-made fascicle or attachment semantics.[^25]

The **Visible Korean** program is especially relevant to facial soft tissues. A published advanced head dataset describes 4,000 sections at 0.04 mm voxel size. Prepared browsing volumes and anatomical surface products also exist, at different resolutions and with different coverage. Their availability and permissions must be checked for the precise product; an interactive PDF or a downsampled NIfTI viewer does not establish access to the original annotated stack.[^26][^27]

The **Chinese Visible Human** work likewise supplies a route to real section-derived geometry and has supported facial-template research. Its primary descriptions do not establish a current, unrestricted download of a complete face with the requested field set. A raw-data acquisition would leave segmentation, annotation, and mechanical interpretation to the project.[^28]

**Recommendation:** use these resources when an anatomy reconstruction project is acceptable. They offer stronger observational foundations than an artist-built mesh, but are unlikely to minimize preprocessing without an existing regional segmentation partnership.

### Regional attachment evidence

Hur and colleagues' 2020 study combines dissection and micro-CT to examine dermal insertions and intermingling of upper-lip/nasal-region muscle fibers around the nasolabial fold. The paper and supporting videos provide strong local evidence, but no complete registered head or machine-readable attachment atlas was verified.[^29]

The 2018 orbicularis retaining ligament micro-CT study describes a branching, multilayered fibroelastic structure linking deeper tissues and dermis. This is relevant to the representation of attachments: a single straight spring or one smooth band may be a substantial simplification even when the anatomical name is correct. Again, the visible reconstructions and supporting media are not a verified reusable whole-head dataset.[^30]

These studies support a focused regional benchmark. For the nasolabial fold, request tissue blocks preserving dermis, the relevant muscle bundles, and intervening connective tissue in the same frame. For the periorbital region, request reconstructed plates or septa and their attachment surfaces. Such data could answer a local mechanics question without first reconstructing every vessel, nerve, or organ in the head.

## Lower-priority and excluded routes

| Route | Assessment for this task |
| --- | --- |
| Facial Tissue Simulator historical project | The project page describes a planned release; a current downloadable anatomy package was not verified. Treat it as a historical research lead.[^31] |
| Phace and generalized learned physical-face models | Valuable approaches to fitting physical deformation or activation. Their learned control fields do not establish specimen-derived muscle fascicles or connective anatomy.[^32][^33] |
| Generic surface rigs and face templates | Useful for identity, expression targets, rendering, or personalization. They should not be counted as internal anatomy upgrades without separate anatomical data. |
| “3D Visible Human geometry” lower-limb datasets | The inspected University of Denver release concerns lower-extremity anatomy. It is not a ready-made facial atlas despite sharing the Visible Human name.[^34] |
| Nonhuman facial DTI / contrast-enhanced CT | Relevant acquisition methods, but a nonhuman specimen is not a human-head replacement. Anatomical transfer would introduce another modeling hypothesis. |
| Toyota THUMS | An obtainable injury-oriented FEM with active musculature, but public scope and validation target crash biomechanics. Detailed facial-expression coverage and the requested facial fields were not established. Free access is through its registration terms, with restricted sharing among registered users.[^45] |
| Anatomy Standard | Useful anatomy and jaw/TMJ references; the checked public products do not offer a complete raw facial-expression dataset with skin, dermal attachments, and fibers.[^46] |
| Complete Anatomy, Visible Body, BioDigital | Interactive anatomy/reference products. A viewing subscription should not be assumed to provide exportable source geometry or FEM reuse rights. Complete Anatomy's published export-use page concerns imagery/video; no suitable public raw-model distribution was verified for these platforms.[^47] |
| Ziva | Unity ended sales/support of the former products in 2024. DNEG now describes proprietary technology and services; no off-the-shelf head anatomy package with the required field contract was verified.[^48] |
| Human Dissection Models, UNAM | Cadaver-dissection-derived educational models are documented, but the checked institutional description does not establish an obtainable complete head with the required fields. Historical educational availability is not a current dataset license.[^49] |
| Faceform Wrap | Registration and topology-transfer software already relevant to the pipeline. It does not itself supply the missing anatomical observations.[^50] |

## Integration and evaluation plan

### Keep evidence, geometry, and mechanics separate

The next model should carry a small, explicit data contract. Each structure needs a stable identifier, anatomical name, side, source/version, units, and reference frame. Each geometry or field should additionally identify whether it was measured, manually segmented, registered, artist-modeled, interpolated, or inferred. Where a transformation combines donors or sources, preserve that transformation and the original data separately.

For muscle architecture, retain bundle trajectories or local direction distributions before reducing them to a single vector per tetrahedron. Perioral interdigitation can make a single direction an inadequate representation. Interpolating a clean-looking field should not erase crossings or turn low-confidence regions into apparent observations.

For attachments, represent receiving tissue and attachment extent explicitly. A conforming connection, embedded bundle, distributed tie, or sliding interface can each be appropriate to a particular model. The data should establish the intended relation; the solver should implement and test it. Adding a second constraint between nodes already shared by skin and volume does not create a new anatomical connection.

### Evaluate one region before replacing the whole head

1. **Inventory and geometry gate.** Compare the candidate manifest against the current 35-component active set. Inspect lips, eyelids, muscle endpoints, and relevant fasciae. Check units, gaps, intersections, thickness, and whether muscle surfaces define usable volumes. Prefer a small facial sample containing skin, bone, muscles, and connective tissue together.
2. **Anatomical evidence gate.** Select either the upper-lip/nasolabial region or the periorbital region. Require enough source information to identify which attachment and fiber features are observed and which remain reconstructed. A named surface without its anatomical continuity does not pass this gate.
3. **Mechanical integration gate.** Demonstrate that changing the imported fiber or attachment data changes the intended force term. For Apple, a fiber-informed test must use a fiber-constrained activation mode or a deliberately modified constitutive model; the current unconstrained tensor modes cannot establish a benefit from loading a fiber array.
4. **Controlled comparison.** Hold target motion, mesh resolution, passive stiffness, boundary conditions, and optimization budget as consistent as practical. Compare baseline, candidate anatomy, and a control with comparable passive reinforcement but altered attachment location/direction. This helps separate an anatomical improvement from a generic stiffness change.
5. **Outcome assessment.** Measure landmark motion and local crease location, depth, and width as well as global fit. A lower displacement RMS alone does not establish better fold anatomy or a correct attachment mechanism. Assess mesh quality and stability separately from biological fidelity.

These are proposed acceptance checks, not completed experiments. The present research did not change model construction, materials, activation, or solver behavior.

### Acquisition questions

The following questions make a supplier or laboratory response technically reviewable:

| Required response | Why it matters |
| --- | --- |
| Exact product/dataset version and a head-only structure manifest | Whole-body object counts can conceal missing facial structures. |
| Sample containing skin, one relevant muscle group, connective tissue, and bone in one frame | A standalone muscle render cannot establish interfaces or alignment. |
| Source of each fascia, aponeurosis, ligament, and insertion region | “Cadaver based” may describe only part of the model. |
| Original fascicle trajectories or numerical direction arrays, with coordinates and method | Normal maps and painted striations cannot replace these fields. |
| Explicit attachment entity types, endpoints/regions, and preserved dermal/mucosal surfaces | Bone origin masks alone do not answer muscle-to-skin coupling. |
| Measured versus sculpted or interpolated structures, donor count, and registration chain | This determines what anatomical claims are justified. |
| Editable formats, scale, object hierarchy, and surface/volume quality report | This determines the actual preprocessing savings. |
| Written terms for FEM conversion, derivative mesh redistribution, paper figures, code/data release, and commercial use if needed | Viewing or animation rights do not necessarily cover the research deliverables. |

**Recommended order:** inspect ArtiSynth and the open surface candidates without procurement; seek Toronto and Park–Noël data terms for the missing anatomy; evaluate an Atomedge facial sample and a comparable Zygote sample if paid geometry remains attractive. Use MIDA when its internal-reference value and licensing fit are clear. Full raw-image reconstruction is the fallback when neither a laboratory nor a supplier can provide the required anatomical evidence.

## Sources

Unless a publication date is stated, product, repository, access, and license information refers to the versions available on 12 September 2026. Local implementation evidence is detailed in the linked pipeline audit. Official vendor descriptions are evidence of advertised deliverables, not independent anatomical validation.

[^1]: Li Z, Tran D, Bibliowicz J, Khan A, Mogk JPM, Agur AM. [High Fidelity 3D Anatomical Visualization of the Fibre Bundles of the Muscles of Facial Expression as In situ](https://link.springer.com/chapter/10.1007/978-3-030-61905-3_10). *Digital Anatomy*, 2021, pp. 185–197. DOI: 10.1007/978-3-030-61905-3_10. Methods and architecture; chapter access does not confer dataset access.
[^2]: Park JS, Noël G. [Facial retaining ligaments based on real color sectioned images with 3D models: Toward a more precise classification](https://www.sciencedirect.com/science/article/pii/S1010518225002252). *Journal of Cranio-Maxillofacial Surgery* 53(9), 2025, pp. 1638–1646. DOI: 10.1016/j.jcms.2025.07.002. [Official publisher metadata](https://api.elsevier.com/content/article/PII:S1010518225002252?httpAccept=text/xml) confirms article open-access status.
[^3]: ArtiSynth. [Models package download](https://www.artisynth.org/Software/ModelsDownload). Public source and release availability; version matching requirements.
[^4]: Atomedge. [Tommy complete male anatomy model](https://www.atomedge.com/3dmodel). Advertised geometry, source inputs, formats, version, and enquiry-based pricing.
[^5]: IT'IS Foundation / FDA. [MIDA model distribution](https://itis.swiss/virtual-population/regional-human-models/mida-model) and [MIDA v1.0 request](https://itis.swiss/virtual-population/regional-human-models/mida-model/mida-v1-0). DOI: 10.13099/ViP-MIDA-V1.0.
[^6]: IT'IS Foundation. [Terms and Conditions of User License MIDA Model v1.0](https://itis.swiss/assets/Downloads/VirtualPopulation/License_Agreements/LicenseAgreementMIDA_2024.pdf), distributed 2024 license file, especially §§2.3.2–2.3.3. The actual license should govern any acquisition.
[^7]: 3D4SCI. [Complete Human Head Anatomy](https://sketchfab.com/3d-models/complete-human-head-anatomy-c240eee6c2824f8cbb105129392711b2). Candidate identification of the existing GLB; acquisition provenance remains unproven.
[^8]: Ghassemi A, Prescher A, Riediger D, Axer H. [Anatomy of the SMAS revisited](https://pubmed.ncbi.nlm.nih.gov/15058546/). *Aesthetic Plastic Surgery* 27, 2003, pp. 258–264. DOI: 10.1007/s00266-003-3065-3.
[^9]: Sandulescu T and colleagues. [Histological, SEM and three-dimensional analysis of the midfacial SMAS—New morphological insights](https://pubmed.ncbi.nlm.nih.gov/30468848/). 2018 online publication. Regional histology and 3D reconstruction.
[^10]: Minelli L, van der Lei B, Mendelson BC. [The Superficial Musculoaponeurotic System: Does It Really Exist as an Anatomical Entity?](https://pmc.ncbi.nlm.nih.gov/articles/PMC11027987/). *Plastic and Reconstructive Surgery* 153(5), 2024, pp. 1023–1034; published online 2023. DOI: 10.1097/PRS.0000000000010557.
[^11]: ArtiSynth contributors. [Models package license](https://github.com/artisynth/artisynth_models/blob/master/LICENSE). Redistribution, copyright, subpackage notices, and academic citation conditions.
[^12]: ArtiSynth contributors. [BadinFaceDemo.java](https://github.com/artisynth/artisynth_models/blob/master/src/artisynth/models/face/BadinFaceDemo.java) and [face geometry directory](https://github.com/artisynth/artisynth_models/tree/master/src/artisynth/models/face/geometry). Meshes, muscle topology, and attachment node sets.
[^13]: ArtiSynth contributors. [BadinFemMuscleFaceDemo.java](https://github.com/artisynth/artisynth_models/blob/master/src/artisynth/models/face/BadinFemMuscleFaceDemo.java). Derivation of muscle elements and direction data from paths.
[^14]: ArtiSynth. [Integrated face–tongue–jaw model](https://www.artisynth.org/Demo/IntegratedFaceTongueJaw) and [FRANK head and neck model](https://www.artisynth.org/Demo/FRANKHeadNeckModel). Demonstration scope; availability of a specific full assembly should be checked separately.
[^15]: University of Toronto Musculoskeletal and Peripheral Nerve Anatomy Laboratory. [Research](https://mskpnanatomy.com/research/); [Anne Agur faculty page](https://surgery.utoronto.ca/faculty/anne-agur). Institutional route for data enquiries.
[^16]: Iacono MI and colleagues. [MIDA: A Multimodal Imaging-Based Detailed Anatomical Model of the Human Head and Neck](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0124126). *PLOS ONE* 10(4), 2015, e0124126. DOI: 10.1371/journal.pone.0124126.
[^17]: Iacono MI and colleagues. [MIDA supplementary segmentation methods, S1](https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0124126.s001). Constructed skin layer and manual/atlas-assisted facial segmentation.
[^18]: Z-Anatomy contributors. [PC-Version repository and license](https://github.com/LluisV/Z-Anatomy/tree/PC-Version); [distributed FBX models](https://github.com/LluisV/Z-Anatomy/tree/PC-Version/Resources/Models/FBX). `MuscularSystem100.fbx` geometry-name inventory inspected directly.
[^19]: BodyParts3D / DBCLS. [Version 3.0 archive](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/), including `parts_list_e.txt`, `README_e.html`, and `release_3.0_e.html`. Facial structures, license, and geometry limitations.
[^20]: OMFAtlas author. [Facial source and registration notes](https://github.com/choxos/OMFAtlas/blob/main/documentation/qa/FACIAL-SOURCE.md) and [repository](https://github.com/choxos/OMFAtlas). Version-specific geometry recovery and attribution.
[^21]: Zhang X and colleagues. [An eFace-Template Method for Efficiently Generating Patient-Specific Anatomically-Detailed Facial Soft Tissue FE Models for Craniomaxillofacial Surgery Simulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC4833683/). *Annals of Biomedical Engineering*, 2016; online 2015. DOI: 10.1007/s10439-015-1480-7.
[^22]: Zhang X, Kim D and colleagues. [An eFTD-VP framework for efficiently generating patient-specific anatomically detailed facial soft tissue FE mesh for craniomaxillofacial surgery simulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5845478/). *Biomechanics and Modeling in Mechanobiology* 17, 2018, pp. 387–402; online 2017. DOI: 10.1007/s10237-017-0967-6.
[^23]: Wu T, Hung AP, Mithraratne K. [Generating facial expressions using an anatomically accurate biomechanical model](https://pubmed.ncbi.nlm.nih.gov/26355331/). DOI: 10.1109/TVCG.2014.2339835. Layered facial mechanics and explicit muscles.
[^24]: Wu T, Hunter P, Mithraratne K. [Simulating and Validating Facial Expressions using an Anatomically Accurate Biomechanical Model Derived from MRI Data: Towards Fast and Realistic Generation of Animated Characters](https://www.scitepress.org/papers/2013/42935/42935.pdf), 2013. MRI/manual construction and symmetry assumptions.
[^25]: US National Library of Medicine. [Visible Human Project](https://www.nlm.nih.gov/research/visible/visible_human.html); [additional head-image archive](https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/index.html). Public-domain source imaging.
[^26]: Chung BS, Han M, Har D, Park JS. [Advanced Sectioned Images of a Cadaver Head with Voxel Size of 0.04 mm](https://www.jkms.org/pdf/10.3346/jkms.2019.34.e218). *Journal of Korean Medical Science* 34, 2019, e218. DOI: 10.3346/jkms.2019.34.e218.
[^27]: Visible Korean project authors. [Visible Korean homepage and distributed anatomical content](https://pmc.ncbi.nlm.nih.gov/articles/PMC5890020/), 2018; [prepared head-volume study](https://www.jkms.org/search.php?code=0063JKMS&id=10.3346%2Fjkms.2019.34.e86&vmode=PUBREADER&where=aview), 2019, DOI: 10.3346/jkms.2019.34.e86. Product and resolution distinctions.
[^28]: Zhang SX and colleagues. [The Chinese Visible Human project](https://pmc.ncbi.nlm.nih.gov/articles/PMC1571260/), 2004. Original section-derived anatomical resource; historical access description is not a current license grant.
[^29]: Hur MS and colleagues. [Heights and spatial relationships of the facial muscles acting on the nasolabial fold by dissection and three-dimensional microcomputed tomography](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237043). *PLOS ONE* 15(8), 2020, e0237043. DOI: 10.1371/journal.pone.0237043.
[^30]: O J, Kwon HJ, Choi YJ and colleagues. [Three-dimensional structure of the orbicularis retaining ligament: an anatomical study using micro-computed tomography](https://pmc.ncbi.nlm.nih.gov/articles/PMC6242969/). *Scientific Reports*, 2018. DOI: 10.1038/s41598-018-35425-0.
[^31]: Sifakis E. [Facial Tissue Simulator project](https://pages.cs.wisc.edu/~sifakis/project_pages/fts.html). Historical planned software/data release; current downloadable model not verified.
[^32]: Ichim AE and colleagues. [Phace: Physics-based Face Modeling and Animation](https://infoscience.epfl.ch/server/api/core/bitstreams/b9c13674-0a33-4c2d-9cbc-d03e62c91a87/content), 2017. Physical face fitting/activation representation.
[^33]: Yang and colleagues / Disney Research. [Learning a Generalized Physical Face Model From Data](https://arxiv.org/abs/2402.19477), 2024. Learned physical-face controls and generalization rather than a measured fascicle atlas.
[^34]: University of Denver. [Three-Dimensional Geometry of the Visible Human](https://digitalcommons.du.edu/visiblehuman/). Inspected release concerns lower-extremity anatomy.
[^35]: Atomedge. [Academic integrity and UCT ethics](https://www.atomedge.com/academic-integrity). Source inputs and ethics approval HREC 595/2015.
[^36]: Atomedge. [Tommy complete male anatomy v2.0 anatomical-element inventory](https://www.atomedge.com/s/Atomedge-Tommy-Complete-Male-Anatomy-V20-List-of-Anatomical-Elements.pdf). Named facial muscle/fascia geometry; version differs from current product-page v3.0.
[^37]: Zygote. [Solid 3D Human Head](https://www.zygote.com/cad-models/solid-3d-human-anatomy/3d-cad-human-head-model). USD 6,825, downloadable CAD formats, advertised head systems.
[^38]: Zygote. [Solid male muscular/skeletal collection](https://www.zygote.com/cad-models/collections-products/solid-3d-male-muscular-skeletal-collection). Muscle construction method and relationship to scanned skeleton.
[^39]: Zygote. [Polygon male muscular system](https://www.zygote.com/poly-models/3d-male-systems/3d-male-muscular-system). Facial structure inventory; separate product from the solid head.
[^40]: Atomedge. [License categories](https://www.atomedge.com/license-plans) and [model enquiry](https://www.atomedge.com/3d-model-enquiry). Scope-dependent quote and research/commercial/publishing distinctions.
[^41]: Ebers / CGTrader. [Facial Anatomy Layers](https://www.cgtrader.com/3d-models/science/medical/facial-anatomy-layers). USD 549 listing, advertised layers, formats, and technical verification.
[^42]: TurboSquid vendor. [Human Head Anatomy Model with Facial Fat Layers, product 2450228](https://www.turbosquid.com/3d-models/human-head-anatomy-model-with-facial-fat-layers-2450228). USD 169 listing; fat-compartment supplement.
[^43]: TurboSquid vendor. [Full Human Head Anatomy, product 1925680](https://www.turbosquid.com/3d-models/full-human-head-anatomy-1925680). USD 99 listing; identification as the active Melon GLB is not established.
[^44]: Zygote. [Terms](https://www.zygote.com/terms). Product files are governed by their accompanying licenses. TurboSquid. [3D model license](https://www.turbosquid.com/licensing). Source-model access and redistribution constraints apply independently of permission to render or simulate.
[^45]: Toyota. [THUMS model scope and specifications](https://www.toyota.co.jp/thums/about); [User Policy v2](https://www.toyota.co.jp/thums/contents/pdf/THUMS_USER_POLICY_Ver2.pdf). Injury modeling, registered access, and sharing conditions.
[^46]: Anatomy Standard. [Skull, Teeth & TMJ](https://www.anatomystandard.com/skull-teeth-tmj/) and [project](https://www.anatomystandard.com/). Reference applications and public scope.
[^47]: Elsevier / Complete Anatomy. [Use of imagery and videos](https://3d4medical.com/use-of-imagery-videos). Export-use scope. [Visible Body](https://www.visiblebody.com/) and [BioDigital](https://www.biodigital.com/) official product sites were screened for a public reusable source-geometry offer.
[^48]: Unity. [Update about Ziva](https://unity.com/blog/news/update-about-ziva), 2 April 2024. DNEG. [Ziva technology and services](https://www.dneg.com/creative-technology/ziva). Current product-status distinction.
[^49]: UNAM Faculty of Medicine. [Human Dissection](https://gaceta.facmed.unam.mx/index.php/2018/01/24/human-dissection/), 24 January 2018. Historical educational project and download limitations.
[^50]: Faceform. [Wrap](https://faceform.com/). Registration/software scope.
