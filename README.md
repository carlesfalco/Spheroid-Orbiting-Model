# Spheroid Mathematical Model

This is the repository for the manuscript "Collective Transitions from Orbiting to Matrix Invasion in 3D Multicellular Spheroids" by Jiwon Kim, Hyuntae Jeong, Carles Falcó, Alex M. Hruska, W. Duncan
Martinson, Alejandro Marzoratti, Mauricio Araiza, Haiqian Yang, Christian Franck, José A. Carrillo, Ming Guo, and Ian Y. Wong. We include code to solve numerically a the mathematical model and to produce the figures in the main text. We employ an active particle-based model in which cells are subject to active-drag, cell-cell, and cell-matrix forces. The basic equations of motion are:\
$$ \frac{\mathrm{d}\mathbf{x}_i}{\mathrm{d} t} & = \mathbf{v}_i\label{eqn:pos}\,,
    \\
    \frac{\mathrm{d}\mathbf{v}_i}{\mathrm{d}t} & = \underbrace{\left(\alpha - \beta|\mathbf{v}_i|^2\right)\mathbf{v}_i}_{\text{active and drag forces}} + \underbrace{\sum_{j\neq i,\,j = 1}^N \mathbf{F}^{cc}(\mathbf{x}_j-\mathbf{x}_i)}_{\text{cell-cell repulsion/attraction}} + \underbrace{\sum_{k = 1}^M \mathbf{F}^{cm}(\mathbf{y}_k- \mathbf{x}_i,{\mathbf{v}}_i)}_{\text{cell-ECM repulsion/attraction}}$$
