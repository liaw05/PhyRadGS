PhyRadGS License
================

The goal of this License is to allow the research community to use, test and
evaluate the *Software*.

## 1. Definitions

*Licensee* means any person or entity that uses the *Software* and distributes
its *Work*.

*Licensor* means the copyright holders and contributors of the *Software*.

*Software* means the original work of authorship made available under this
License, i.e. PhyRadGS.

*Work* means the *Software* and any additions to or derivative works of the
*Software* that are made available under this License.

## 2. Purpose

This license is intended to define the rights granted to the *Licensee* by
Licensors under the *Software*.

## 3. Rights granted

Licensors grant non-exclusive rights to use the *Software* for research purposes
to research users (both academic and industrial), free of charge, without right
to sublicense. The *Software* may be used "non-commercially", i.e., for research
and/or evaluation purposes only.

Subject to the terms and conditions of this License, you are granted a
non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
publicly display, publicly perform and distribute its *Work* and any resulting
derivative works in any form.

## 4. Limitations

**4.1 Redistribution.** You may reproduce or distribute the *Work* only if (a) you do
so under this License, (b) you include a complete copy of this License with
your distribution, and (c) you retain without modification any copyright,
patent, trademark, or attribution notices that are present in the *Work*.

**4.2 Derivative Works.** You may specify that additional or different terms apply
to the use, reproduction, and distribution of your derivative works of the *Work*
("Your Terms") only if (a) Your Terms provide that the use limitation in
Section 2 applies to your derivative works, and (b) you identify the specific
derivative works that are subject to Your Terms. Notwithstanding Your Terms,
this License (including the redistribution requirements in Section 3.1) will
continue to apply to the *Work* itself.

**4.3** Any other use without prior consent of Licensors is prohibited. Research
users explicitly acknowledge having received from Licensors all information
allowing to appreciate the adequacy between the *Software* and their needs and
to undertake all necessary precautions for its execution and use.

**4.4** In case of using the *Software* for a publication or other results obtained
through the use of the *Software*, users are strongly encouraged to cite the
corresponding publications as explained in the documentation of the *Software*.

## 5. Disclaimer

THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE *SOFTWARE* FOR COMMERCIAL PURPOSES
WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS. ANY SUCH ACTION WILL
CONSTITUTE A FORGERY. THIS *SOFTWARE* IS PROVIDED "AS IS" WITHOUT ANY WARRANTIES
OF ANY NATURE AND ANY EXPRESS OR IMPLIED WARRANTIES, WITH REGARDS TO COMMERCIAL
USE, PROFESSIONAL USE, LEGAL OR NOT, OR OTHER, OR COMMERCIALISATION OR
ADAPTATION. UNLESS EXPLICITLY PROVIDED BY LAW, IN NO EVENT SHALL THE LICENSOR
OR THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES, LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING FROM, OUT OF OR
IN CONNECTION WITH THE *SOFTWARE* OR THE USE OR OTHER DEALINGS IN THE *SOFTWARE*.

## 6. Third-Party Components

This Software includes components under separate licenses:

**gsplat** (`algorithms/gsplat_xray/gsplat_xray/`)
  - Licensed under Apache License 2.0
  - Copyright (c) The Regents of the University of California, Nerfstudio Team
  - See https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE

**R2-Gaussian** (`algorithms/r2_gaussian/`)
  - Licensed under the Gaussian-Splatting License
  - Copyright Inria and Max Planck Institut for Informatik (MPII)
  - See algorithms/r2_gaussian/LICENSE.md

**simple-knn** (`algorithms/submodules/simple-knn/`)
  - Licensed under the Gaussian-Splatting License
  - See https://gitlab.inria.fr/bkerbl/simple-knn

**pytorch-ssim** (included in `utils/loss_utils.py`)
  - Licensed under MIT License
  - Copyright Evan Su, 2017
  - See https://github.com/Po-Hsun-Su/pytorch-ssim