# SCGSR Mid-Award Progress Report: Narrative (form-ready text)

*Plain text for pasting into the web-form fields. No figures or equations (those live in the separate figures PDF). Acronyms are spelled out on first use. Figures are referred to as "Figure N".*

---

## 1. Research Goals and Objectives

**Overall goal.** This project develops a multi-modal Foundation Model (a single large neural network, pre-trained on broad data, that can be adapted to many downstream tasks) for particle reconstruction in Liquid Argon Time Projection Chambers (LArTPCs). LArTPCs are the detector technology of the Short-Baseline Neutrino (SBN) program at Fermi National Accelerator Laboratory and of the future Deep Underground Neutrino Experiment (DUNE). Conventional reconstruction pipelines process the raw detector data through several approximate stages before pattern recognition. Instead, this project learns directly from the two lowest-level detector data streams: the ionization charge recorded on the Time Projection Chamber (TPC) wire planes, and the scintillation light recorded by the Photon Detection System (PDS), so that the model can implicitly learn the detector response and reduce detector-related systematic uncertainties.

**Objectives (as proposed).**
- **Aim 1 (Architecture).** Develop a scalable multi-modal Foundation Model able to process the high-dimensional, sparse raw waveform data from both the TPC wire planes and the PDS.
- **Aim 2 (Training).** Pre-train the model with self-supervised learning (learning structure from unlabeled data), using cross-modal masked-autoencoding, on a large simulated dataset.
- **Aim 3 (Evaluation).** Evaluate the pre-trained model on downstream tasks relevant to neutrino physics analysis (particle classification and energy estimation) using fine-tuning or linear probes, and compare against state-of-the-art machine-learning reconstruction methods (such as SPINE and PoLAr-MAE) that operate on conventionally pre-processed data, to quantify the improvement in reconstruction and the implicit learning of detector effects.

**Technical approach.** Rather than reconstructing tracks from pre-processed inputs, the model is trained on the raw digitized waveforms of both modalities, generated from the same simulated interactions. The strategy is directly inspired by modern audio machine learning, where a raw audio waveform is not fed to a model directly but is first converted into a compact, multi-scale representation; the detector data are time-series waveforms of the same character. A prerequisite for training, addressed during this reporting period, is therefore a signal-representation pipeline that converts the raw, noise-dominated waveforms into a compact, denoised form a neural network can ingest efficiently at the scale of the full dataset. Establishing that this is possible removes the central feasibility risk of the project before large-scale computation is committed to training the model.

**Contribution to thesis.** The work supports my doctoral research on data-driven modeling of neutrino-argon interactions, where reducing detector-related systematic uncertainties is a key enabler.

**Milestones (award period: January–December 2026).** The proposed timeline had three phases; status at the mid-award point (July 2026):
- **Months 1–2 (setup and finalization of the simulated dataset): completed (100%).**
- **Months 3–8 (architecture development and self-supervised pre-training): in progress (~40%).** The data representation on which the architecture operates is complete and validated; implementation and pre-training of the model is the active next phase.
- **Months 9–12 (evaluation on downstream tasks): not yet started (0%), on schedule.**

The project is on plan against the proposed schedule, and the goals are unchanged from the proposal.

---

## 2. Project Progress and Accomplishments

The reporting period established the two foundations the Foundation Model requires before training can begin: a large, realistic, multi-modal simulated dataset, and a signal-representation pipeline that turns the raw detector waveforms into a compact, denoised form a neural network can consume. The raw waveforms cannot be used directly, for two reasons: they are dominated by detector noise, and their raw size (millions of samples per event across both modalities) is far too large for a transformer, whose computational cost grows sharply with the number of inputs. The front-end therefore has two jobs. The first is denoising: removing the detector noise so that the model learns from the physics signal rather than from the electronics noise. The second is sparsification: re-expressing each waveform as a small set of wavelet coefficients, a compact and structured list of numbers, so that an event becomes a manageable number of inputs, or "tokens," for the model. Both jobs are accomplished with a discrete wavelet transform (DWT), a standard multi-scale signal-decomposition method: the transform decomposes each waveform into components across a range of time scales (a discrete wavelet decomposition), and discarding the components that are consistent with noise is exactly what leaves a small, physics-bearing set of coefficients behind. All processing was implemented in a dedicated software toolkit (named HELIX) developed during this period.

**Major activities and key results.**

**(a) A large-scale, paired, multi-modal simulated dataset.** We produced a simulated dataset ("Doraemon") of approximately one million detector events for a generic SBN-like LArTPC. Crucially, the charge and light are generated from the *same* underlying particle interaction, so the two modalities are physically paired event-by-event, a property essential for cross-modal learning and impossible to obtain from real data. This is made possible by a Graphics Processing Unit (GPU)-based simulation that produces a full detector realization in about one second per event. The dataset is stored at every level of the physics chain (truth energy deposits; detector signals; digitized raw waveforms; and the denoised representation), so the model can be trained on the rawest possible input while being supervised against exact simulation truth.

**(b) High-fidelity detector response and noise.** The simulation was upgraded to a high-fidelity noise model tuned to match MicroBooNE, the most thoroughly characterized SBN detector. It contains two components. The dominant and most difficult one is coherent noise: a correlated disturbance shared across whole groups of neighboring wires that is large in amplitude and, critically, occupies the same slow time scales as the physics signal, so it cannot be removed by a simple frequency filter without also damaging the signal. The second is independent, per-wire electronics noise. Coherent noise is the principal obstacle in real LArTPC signal processing, and reproducing it faithfully is what makes the denoising problem in simulation representative of real detector data.

**(c) The denoising-and-compression pipeline.** We built and validated the pipeline that performs both jobs. It first removes the coherent noise, using a method that separates the correlated common disturbance from the sparse physics signal scale by scale, and then sparsifies each waveform with the discrete wavelet transform, keeping only the coefficients that stand above the remaining noise. On the TPC wire planes, which carry the bulk of the event information and are the more important case, the reconstruction recovers the true charge with a fidelity of 0.91–0.96 (out of 1.0), is unbiased across three orders of magnitude in signal size, and reduces the noise away from tracks essentially to zero, while compressing each plane by roughly 170 to 190 times (Figures 1–3). On the light detectors, the waveforms compress by about 30 times, itself a large reduction, set by the point where the reconstruction error reaches the intrinsic noise floor (Figures 4–5). In both cases the size of the compact representation scales with the physics content of an event rather than its raw size, which is exactly what a neural network needs.

**(d) A GPU processing pipeline.** The full chain (noise generation, coherent-noise removal, and the wavelet transform) was ported to run on GPU, yielding roughly a 30-times speedup and making processing at dataset scale practical.

**Challenges and negative findings.**
- **Data at scale.** The datasets are multi-terabyte, and reading and indexing them over the shared filesystem is a genuine bottleneck; initial loads were slow and required restructuring the data access (subsetting and caching) to be workable.
- **Compute access and setup at scale.** Obtaining and configuring GPU compute (software containers, allocations, and getting multiple numerical frameworks to run together) took non-trivial time and is a gating factor for scaling up to model training.
- **An honest finding: compression is set by information content.** How much a waveform can be compressed is governed by how much real information it carries, not by the method. The largely empty wire planes compress by up to two orders of magnitude; the information-rich scintillation pulse still compresses by a substantial factor of about 30. In every case the gains come from removing noise, not from discarding signal, which is precisely the property required for a model that must learn from the physics.

**Status against goals.** The dataset and the representation pipeline (the prerequisites under Aims 1 and 2) are complete and validated at the scale of the full dataset. Establishing and validating this representation was a necessary de-risking step before committing large-scale computation to model training, and it is what makes that training tractable now. Implementation and self-supervised pre-training of the model is the active next phase; the downstream evaluation (Aim 3) has not yet started and remains scheduled for the final period. Full model results will be reported in the final report.

---

## 4. Use of DOE Laboratory Facilities and Research Capabilities

This research used the Shared Scientific Data Facility (S3DF) at SLAC National Accelerator Laboratory. Its GPU compute was essential for the detector simulation and for the denoising-and-compression pipeline, and its large-scale storage was essential for holding and serving the multi-terabyte simulated datasets.

---

## 5. Plans for the Remaining Period

For the remainder of the award (July–December 2026), the plan is to (1) finalize the data loader that streams the compact, paired representation for both detector modalities at scale; (2) implement and pre-train the multi-modal Foundation Model using self-supervised (masked-autoencoding) learning on this representation; and (3) evaluate it on the downstream tasks of particle classification and energy estimation, using fine-tuning or linear probes, against state-of-the-art machine-learning reconstruction methods such as SPINE and PoLAr-MAE that operate on pre-processed data. Pre-training the model is the primary objective of this phase, with the downstream evaluation following as the model matures; the depth of that evaluation will depend on the time remaining in the award. The foundation completed this period, which de-risks the central feasibility question of whether a neural network can consume raw LArTPC data at scale, positions the project to focus the final months on model training and evaluation. The model architecture, training, and evaluation results will be the subject of the final report.

---

## 6. Reporting Changes or Issues

No. I do not foresee changes or issues in accomplishing the research goals and objectives.
