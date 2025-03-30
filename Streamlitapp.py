import streamlit as st
import os
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from PIL import Image
import base64

# Set page config
st.set_page_config(
    page_title="Advanced Well Log Analyzer",
    page_icon="🛢️",
    layout="wide"
)

# Background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
            background-attachment: fixed;
            background-opacity: 0.1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background (make sure to have 'oil_rig.jpg' in your directory)
if os.path.exists("oil_rig.jpg"):
    add_bg_from_local("oil_rig.jpg")

# Initialize session state
if 'las_data' not in st.session_state:
    st.session_state.las_data = None
    st.session_state.well_data = None
    st.session_state.curve_names = []
    st.session_state.header_info = {}
    st.session_state.formation_tops = {}
    st.session_state.lithology_names = {0: 'Shale', 1: 'Sandstone', 2: 'Limestone', 
                                     3: 'Dolomite', 4: 'Anhydrite', 5: 'Salt'}
    st.session_state.plot_colors = {
        'GR': '#00AA00',
        'RT': '#0000FF',
        'RHOB': '#FF0000',
        'NPHI': '#800080',
        'VSH': '#A52A2A',
        'PHI': '#0000FF',
        'SW': '#00008B',
        'background': '#FFFFFF',
        'grid': '#D3D3D3',
        'reservoir': '#00AA00',
        'non_reservoir': '#FF0000',
        'QC_pass': '#00AA00',
        'QC_fail': '#FF0000',
        'QC_warning': '#FFA500'
    }
    st.session_state.archie_params = {
        'a': 1.0,
        'm': 2.0,
        'n': 2.0,
        'Rw': 0.1
    }
    st.session_state.calculations = {
        'VSH_calculated': False,
        'PHI_calculated': False,
        'PHI_EFF_calculated': False,
        'SW_calculated': False
    }

# Helper functions
def get_units(mnemonic):
    """Get units for a curve from LAS header"""
    if st.session_state.las_data:
        for curve in st.session_state.las_data.curves:
            if curve.mnemonic == mnemonic:
                return curve.unit if curve.unit else '-'
    return '-'

def find_matching_curve(options):
    """Helper to find matching curve from options"""
    for opt in options:
        if opt in st.session_state.curve_names:
            return opt
    return None

def add_formation_tops(ax, min_depth, max_depth):
    """Add formation tops to plots"""
    if hasattr(st.session_state, 'formation_tops'):
        for name, depth in st.session_state.formation_tops.items():
            if min_depth <= depth <= max_depth:
                ax.axhline(y=depth, color='black', linestyle='--', linewidth=0.5)
                ax.text(ax.get_xlim()[0], depth, f" {name}", 
                       va='bottom', ha='left', fontsize=8, 
                       backgroundcolor='white', bbox=dict(facecolor='white', alpha=0.7))

def maintain_calculations():
    """Ensure calculations dictionary exists in session state"""
    if 'calculations' not in st.session_state:
        st.session_state.calculations = {
            'VSH_calculated': False,
            'PHI_calculated': False,
            'PHI_EFF_calculated': False,
            'SW_calculated': False
        }

# Main app
st.title("🛢️ Advanced Well Log Analyzer")

# Sidebar for file upload
with st.sidebar:
    st.header("File Operations")
    uploaded_file = st.file_uploader("Upload LAS File", type=['las'])
    
    if uploaded_file is not None:
        try:
            # Read the file content as bytes and pass to lasio
            file_content = uploaded_file.read()
            
            # Handle both string and bytes input for lasio
            try:
                st.session_state.las_data = lasio.read(file_content.decode('utf-8'))
            except:
                # If decode fails, try reading directly (some LAS files might need this)
                uploaded_file.seek(0)  # Reset file pointer
                st.session_state.las_data = lasio.read(uploaded_file)
            
            st.session_state.well_data = st.session_state.las_data.df()
            st.session_state.well_data.reset_index(inplace=True)
            st.session_state.curve_names = [curve.mnemonic for curve in st.session_state.las_data.curves]
            
            # Extract formation tops from LAS file
            st.session_state.formation_tops = {}
            if hasattr(st.session_state.las_data, 'other'):
                for item in st.session_state.las_data.other:
                    if isinstance(item, str) and ('TOP' in item.upper() or 'FORMATION' in item.upper()):
                        parts = item.split()
                        if len(parts) >= 2:
                            try:
                                depth = float(parts[-1])
                                name = ' '.join(parts[:-1])
                                st.session_state.formation_tops[name] = depth
                            except ValueError:
                                continue
            
            st.success(f"Successfully loaded {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Failed to load file: {str(e)}")

# Create tabs
tabs = ["📁 File & Info", "📊 Visualization", "🪨 Lithofacies", "🔍 Formation Evaluation", 
        "📈 Statistics", "🛢️ Reservoir Analysis", "✅ Quality Control", "⚙️ Settings"]
selected_tab = st.radio("Navigation", tabs, horizontal=True)

if st.session_state.las_data is None and selected_tab != "📁 File & Info":
    st.warning("Please upload a LAS file first")
    st.stop()

# File & Info tab
if selected_tab == "📁 File & Info":
    st.header("📁 LAS File Information")
    
    if st.session_state.las_data:
        tab1, tab2, tab3, tab4 = st.tabs(["Well Information", "Curve Information", "Parameters", "Formation Tops"])
        
        with tab1:
            well_info = []
            for item in st.session_state.las_data.well:
                well_info.append({
                    "Property": str(item.mnemonic),
                    "Value": str(item.value)
                })
            st.table(pd.DataFrame(well_info))
        
        with tab2:
            curve_info = []
            for curve in st.session_state.las_data.curves:
                curve_info.append({
                    "Mnemonic": curve.mnemonic,
                    "Description": getattr(curve, 'descr', ''),
                    "Units": getattr(curve, 'unit', ''),
                    "API Code": getattr(curve, 'api_code', 'N/A')
                })
            st.table(pd.DataFrame(curve_info))
        
        with tab3:
            param_info = []
            for param in st.session_state.las_data.params:
                param_info.append({
                    "Parameter": param.mnemonic,
                    "Value": str(param.value)
                })
            st.table(pd.DataFrame(param_info))
        
        with tab4:
            formation_info = []
            if hasattr(st.session_state, 'formation_tops'):
                for name, depth in st.session_state.formation_tops.items():
                    formation_info.append({
                        "Formation": name,
                        "Depth": depth
                    })
            st.table(pd.DataFrame(formation_info))
    else:
        st.info("Upload a LAS file to view information")

# Visualization tab
elif selected_tab == "📊 Visualization":
    st.header("📊 Data Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        depth_min = st.number_input("Min Depth", value=float(st.session_state.well_data[st.session_state.curve_names[0]].min()))
    with col2:
        depth_max = st.number_input("Max Depth", value=float(st.session_state.well_data[st.session_state.curve_names[0]].max()))
    
    plot_type = st.selectbox("Plot Type", ["Linear Plot", "Multi-Track", "Cross Plot", "Histogram", "Triple Combo"])
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=st.session_state.plot_colors['background'])
    
    depth_col = st.session_state.curve_names[0]
    data = st.session_state.well_data[(st.session_state.well_data[depth_col] >= depth_min) & 
                                    (st.session_state.well_data[depth_col] <= depth_max)]
    
    if plot_type == "Linear Plot":
        log_col = st.selectbox("Select Log", st.session_state.curve_names)
        
        ax.plot(data[log_col], data[depth_col], linewidth=0.5, color=st.session_state.plot_colors.get(log_col, 'blue'))
        ax.set_xlabel(f"{log_col} ({get_units(log_col)})")
        ax.set_ylabel(f"Depth ({get_units(depth_col)})")
        ax.set_title(f"{log_col} Log")
        ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
        ax.invert_yaxis()
        add_formation_tops(ax, depth_min, depth_max)
    
    elif plot_type == "Multi-Track":
        cols = st.columns(3)
        tracks = []
        for i in range(3):
            with cols[i]:
                tracks.append(st.selectbox(f"Track {i+1}", st.session_state.curve_names, key=f"track_{i}"))
        
        gs = GridSpec(1, len(tracks), width_ratios=[1]*len(tracks))
        axes = []
        
        for i, track in enumerate(tracks):
            ax = fig.add_subplot(gs[i])
            color = st.session_state.plot_colors.get(track, ['green', 'blue', 'red'][i % 3])
            ax.plot(data[track], data[depth_col], linewidth=0.5, color=color)
            ax.set_xlabel(f"{track} ({get_units(track)})")
            ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            if i > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f"Depth ({get_units(depth_col)})")
            ax.invert_yaxis()
            axes.append(ax)
        
        add_formation_tops(axes[0], depth_min, depth_max)
        fig.tight_layout()
    
    elif plot_type == "Cross Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X Axis", st.session_state.curve_names)
        with col2:
            y_col = st.selectbox("Y Axis", st.session_state.curve_names)
        
        ax.scatter(data[x_col], data[y_col], alpha=0.5, color=st.session_state.plot_colors.get('cross_x', 'blue'))
        ax.set_xlabel(f"{x_col} ({get_units(x_col)})")
        ax.set_ylabel(f"{y_col} ({get_units(y_col)})")
        ax.set_title(f"{y_col} vs {x_col} Cross Plot")
        ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
    
    elif plot_type == "Histogram":
        log_col = st.selectbox("Select Log", st.session_state.curve_names)
        
        ax.hist(data[log_col].dropna(), bins=30, edgecolor='black', color=st.session_state.plot_colors.get(log_col, 'blue'))
        ax.set_xlabel(f"{log_col} ({get_units(log_col)})")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{log_col} Histogram")
        ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
    
    elif plot_type == "Triple Combo":
        gr_col = find_matching_curve(['GR', 'GRC', 'GAM'])
        res_col = find_matching_curve(['RT', 'RES', 'ILD'])
        por_col = find_matching_curve(['PHI', 'POR', 'NPHI', 'DPHI'])
        
        if not all([gr_col, res_col, por_col]):
            st.warning("Could not find all required curves for triple combo")
        else:
            gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
            
            # GR track
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data[gr_col], data[depth_col], color=st.session_state.plot_colors.get(gr_col, 'green'), linewidth=0.5)
            ax1.set_xlabel(f"{gr_col} ({get_units(gr_col)})")
            ax1.set_ylabel(f"Depth ({get_units(depth_col)})")
            ax1.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            ax1.invert_yaxis()
            
            # Resistivity track (log scale)
            ax2 = fig.add_subplot(gs[1])
            ax2.semilogx(data[res_col], data[depth_col], color=st.session_state.plot_colors.get(res_col, 'blue'), linewidth=0.5)
            ax2.set_xlabel(f"{res_col} ({get_units(res_col)})")
            ax2.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            ax2.set_yticklabels([])
            ax2.invert_yaxis()
            
            # Porosity track
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data[por_col], data[depth_col], color=st.session_state.plot_colors.get(por_col, 'red'), linewidth=0.5)
            ax3.set_xlabel(f"{por_col} ({get_units(por_col)})")
            ax3.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            ax3.set_yticklabels([])
            ax3.invert_yaxis()
            
            add_formation_tops(ax1, depth_min, depth_max)
            fig.suptitle("Triple Combo Plot")
            fig.tight_layout()
    
    st.pyplot(fig)

# Lithofacies tab
elif selected_tab == "🪨 Lithofacies":
    st.header("🪨 Lithofacies Identification")
    
    cols = st.columns(3)
    features = []
    for i in range(3):
        with cols[i]:
            features.append(st.selectbox(f"Feature {i+1}", st.session_state.curve_names, key=f"feature_{i}"))
    
    n_clusters = st.slider("Number of Facies", 2, 6, 4)
    
    if st.button("Run Clustering"):
        try:
            # Prepare data
            depth_col = st.session_state.curve_names[0]
            X = st.session_state.well_data[features].dropna().values
            X = StandardScaler().fit_transform(X)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            score = silhouette_score(X, labels)
            
            # Add lithology labels to dataframe
            st.session_state.well_data['LITHOLOGY'] = np.nan
            valid_indices = st.session_state.well_data[features].dropna().index
            st.session_state.well_data.loc[valid_indices, 'LITHOLOGY'] = labels
            
            # Update lithology legend
            legend_text = "Lithology Classes:\n"
            for i in range(n_clusters):
                legend_text += f"{i}: {st.session_state.lithology_names.get(i, f'Facies {i}')}\n"
            
            # Plot results
            fig = plt.figure(figsize=(12, 8), facecolor=st.session_state.plot_colors['background'])
            
            # Depth track
            ax1 = fig.add_subplot(131)
            ax1.set_ylabel(f"Depth ({get_units(depth_col)})")
            ax1.set_xticks([])
            ax1.invert_yaxis()
            
            # Lithology track
            ax2 = fig.add_subplot(132)
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            for i in range(n_clusters):
                lith_data = st.session_state.well_data[st.session_state.well_data['LITHOLOGY'] == i]
                if not lith_data.empty:
                    ax2.scatter([0.5]*len(lith_data), lith_data[depth_col], 
                               color=colors[i], s=10, label=st.session_state.lithology_names.get(i, f'Facies {i}'))
            
            ax2.set_xlim(0, 1)
            ax2.set_xticks([])
            ax2.set_title("Lithofacies")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.invert_yaxis()
            
            # Feature plot
            if len(features) >= 2:
                ax3 = fig.add_subplot(133)
                for i in range(n_clusters):
                    cluster_data = X[labels == i]
                    ax3.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                               color=colors[i], label=f'Facies {i}', alpha=0.6)
                
                ax3.set_xlabel(features[0])
                ax3.set_ylabel(features[1])
                ax3.set_title(f"Feature Space (Score: {score:.2f})")
                ax3.grid(True)
            
            fig.tight_layout()
            st.pyplot(fig)
            
            with st.expander("Lithology Legend"):
                st.text(legend_text)
            
            st.success(f"Clustering completed with {n_clusters} facies (Silhouette Score: {score:.2f})")
            
        except Exception as e:
            st.error(f"Clustering error: {str(e)}")

# Formation Evaluation tab
elif selected_tab == "🔍 Formation Evaluation":
    st.header("🔍 Formation Evaluation")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Vshale", "Porosity", "Water Saturation", "Reservoir Flag", "Net Pay"])
    
    with tab1:
        st.subheader("Vshale Calculation")
        gamma_col = st.selectbox("Gamma Ray Log", st.session_state.curve_names, 
                                key='gamma_vsh', index=st.session_state.curve_names.index('GR') if 'GR' in st.session_state.curve_names else 0)
        vsh_method = st.selectbox("Method", ["Linear", "Clavier", "Larionov (Tertiary)", "Larionov (Old)", "Steiber"])
        
        if st.button("Calculate Vshale"):
            gr = st.session_state.well_data[gamma_col]
            gr_clean = st.session_state.well_data[gamma_col].quantile(0.1)
            gr_shale = st.session_state.well_data[gamma_col].quantile(0.9)
            
            if vsh_method == "Linear":
                vsh = (gr - gr_clean) / (gr_shale - gr_clean)
            elif vsh_method == "Clavier":
                vsh = 1.7 - (3.38 - (gr - gr_clean) / (gr_shale - gr_clean + 0.7))**0.5
            elif vsh_method == "Larionov (Tertiary)":
                vsh = 0.083 * (2**(3.7 * (gr - gr_clean) / (gr_shale - gr_clean)) - 0.083)
            elif vsh_method == "Larionov (Old)":
                vsh = 0.33 * (2**(2 * (gr - gr_clean) / (gr_shale - gr_clean)) - 0.33)
            elif vsh_method == "Steiber":
                vsh = 0.5 * (gr - gr_clean) / (gr_shale - gr_clean + 0.5)
            
            st.session_state.well_data['VSH'] = vsh.clip(0, 1)
            st.session_state.calculations['VSH_calculated'] = True
            
            # Plot results
            fig = plt.figure(figsize=(8, 10), facecolor=st.session_state.plot_colors['background'])
            ax = fig.add_subplot(111)
            
            ax.plot(st.session_state.well_data['VSH'], st.session_state.well_data[st.session_state.curve_names[0]], 
                   color=st.session_state.plot_colors.get('VSH', 'brown'), linewidth=0.5)
            ax.set_xlabel('Vshale (v/v)')
            ax.set_ylabel(f"Depth ({get_units(st.session_state.curve_names[0])})")
            ax.set_title(f"Vshale ({vsh_method} Method)")
            ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            ax.invert_yaxis()
            
            st.pyplot(fig)
            st.success(f"Vshale calculated using {vsh_method} method")
    with tab2:
      st.subheader("Porosity Calculation")
    
    # Check if we have both neutron and density logs
    has_neutron = any(log in st.session_state.curve_names for log in ['NPHI', 'NEUT', 'PHIN'])
    has_density = any(log in st.session_state.curve_names for log in ['RHOB', 'DEN', 'DPHI'])
    
    if has_neutron and has_density:
        # Auto-detect neutron and density logs
        neutron_col = find_matching_curve(['NPHI', 'NEUT', 'PHIN'])
        density_col = find_matching_curve(['RHOB', 'DEN', 'DPHI'])
        
        st.info(f"Using {neutron_col} as neutron porosity and {density_col} as density log")
        
        fluid_type = st.selectbox("Fluid Type", ["Oil", "Gas"], key='fluid_type')
        
        # Get matrix parameters based on lithology if available
        matrix_density = 2.65  # Default for sandstone
        matrix_neutron = 0  # Default for sandstone
        
        if 'LITHOLOGY' in st.session_state.well_data.columns:
            matrix_density = np.where(
                st.session_state.well_data['LITHOLOGY'] == 2, 2.71,  # Limestone
                np.where(st.session_state.well_data['LITHOLOGY'] == 3, 2.87,  # Dolomite
                        matrix_density))
            
            matrix_neutron = np.where(
                st.session_state.well_data['LITHOLOGY'] == 2, -0.01,  # Limestone
                np.where(st.session_state.well_data['LITHOLOGY'] == 3, 0.01,  # Dolomite
                        0))  # Sandstone
        
        # Fluid parameters
        if fluid_type == "Oil":
            fluid_density = 1.0  # g/cc
            fluid_neutron = 1.0  # Assuming water-filled
        else:  # Gas
            fluid_density = 0.7  # g/cc
            fluid_neutron = 0.7  # Gas correction
        
        if st.button("Calculate Total Porosity (Neutron-Density Combination)"):
            try:
                # First validate the logs
                if neutron_col not in st.session_state.well_data.columns or density_col not in st.session_state.well_data.columns:
                    raise ValueError("Required logs not found in data")
                
                if st.session_state.well_data[neutron_col].isnull().all() or st.session_state.well_data[density_col].isnull().all():
                    raise ValueError("Logs contain no valid values")
                
                # Calculate individual porosities
                phi_d = (matrix_density - st.session_state.well_data[density_col]) / (matrix_density - fluid_density)
                phi_n = (st.session_state.well_data[neutron_col] - matrix_neutron) / (fluid_neutron - matrix_neutron)
                
                # Convert neutron porosity from fraction to percentage if needed
                if st.session_state.well_data[neutron_col].max() <= 1:  # Assume it's already in fraction
                    phi_n = phi_n * 100  # Convert to percentage for the calculation
                
                # Apply different combination formulas based on fluid type
                if fluid_type == "Gas":
                    # Gas-bearing formation formula: square root of average of squares
                    phi_total = np.sqrt((phi_n**2 + phi_d**2) / 2)
                else:
                    # Oil-bearing formation formula: simple average
                    phi_total = (phi_n + phi_d) / 2
                
                # Store results with proper clipping and conversion
                st.session_state.well_data['PHI_TOTAL'] = (phi_total / 100).clip(0, 0.4)  # Convert back to fraction
                st.session_state.calculations['PHI_calculated'] = True
                
                # Calculate effective porosity only if VSH exists
                if 'VSH' in st.session_state.well_data.columns:
                    st.session_state.well_data['PHI_EFF'] = st.session_state.well_data['PHI_TOTAL'] * (1 - st.session_state.well_data['VSH'])
                    st.session_state.calculations['PHI_EFF_calculated'] = True
                else:
                    # If VSH not available, set PHI_EFF equal to PHI_TOTAL
                    st.session_state.well_data['PHI_EFF'] = st.session_state.well_data['PHI_TOTAL']
                    st.session_state.calculations['PHI_EFF_calculated'] = True
                    st.warning("VSH not calculated - using total porosity as effective porosity")
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 8), facecolor=st.session_state.plot_colors['background'])
                
                # Plot individual porosities
                ax.plot(phi_d / 100, st.session_state.well_data[st.session_state.curve_names[0]], 
                        color='red', linewidth=0.5, label='Density Porosity')
                ax.plot(phi_n / 100, st.session_state.well_data[st.session_state.curve_names[0]], 
                        color='blue', linewidth=0.5, label='Neutron Porosity')
                
                # Plot combined porosity
                ax.plot(st.session_state.well_data['PHI_TOTAL'], st.session_state.well_data[st.session_state.curve_names[0]], 
                        color='green', linewidth=1.0, label='Total Porosity')
                
                ax.plot(st.session_state.well_data['PHI_EFF'], st.session_state.well_data[st.session_state.curve_names[0]], 
                        color='purple', linewidth=1.0, label='Effective Porosity')
                
                ax.set_xlabel('Porosity (v/v)')
                ax.set_ylabel(f"Depth ({get_units(st.session_state.curve_names[0])})")
                ax.set_title(f"Total Porosity ({fluid_type} bearing)\nN-D Combination Method")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
                ax.invert_yaxis()
                
                st.pyplot(fig)
                
                # Show calculation summary
                st.success(f"Total porosity calculated using Neutron-Density combination for {fluid_type} bearing formation")
                st.write("**Calculation Parameters:**")
                st.write(f"- Neutron log: {neutron_col}")
                st.write(f"- Density log: {density_col}")
                st.write(f"- Matrix density: {matrix_density[0] if isinstance(matrix_density, np.ndarray) else matrix_density:.2f} g/cc")
                st.write(f"- Fluid density: {fluid_density:.2f} g/cc")
                st.write(f"- Formula used: {'√[(ΦN² + ΦD²)/2]' if fluid_type == 'Gas' else '(ΦN + ΦD)/2'}")
                
            except Exception as e:
                st.error(f"Error calculating porosity: {str(e)}")
                st.write("**Troubleshooting Tips:**")
                st.write("- Make sure both neutron and density logs are loaded")
                st.write("- Check for null/missing values in the logs")
                st.write("- Verify the units of the neutron log (should be in decimal or percentage)")
                st.write("- Calculate Vshale first for more accurate effective porosity")
    
    else:
        st.warning("Both neutron and density logs are required for N-D combination method")
        st.write("Available methods with single logs:")
        
        porosity_col = st.selectbox("Select Log", st.session_state.curve_names, 
                                  key='porosity_log', index=st.session_state.curve_names.index('RHOB') if 'RHOB' in st.session_state.curve_names else 0)
        porosity_type = st.selectbox("Type", ["Density", "Neutron", "Sonic"])
        fluid_type = st.selectbox("Fluid Type", ["Oil", "Gas"], key='fluid_type_single')
        
        if st.button("Calculate Porosity (Single Log)"):
            try:
                # Validate the selected log
                if porosity_col not in st.session_state.well_data.columns:
                    raise ValueError("Selected log not found in data")
                
                if st.session_state.well_data[porosity_col].isnull().all():
                    raise ValueError("Log contains no valid values")
                
                if porosity_type == "Density":
                    # Default matrix density for sandstone (g/cc)
                    matrix_density = 2.65  
                    
                    # Adjust fluid density based on fluid type
                    fluid_density = 1.0 if fluid_type == "Oil" else 0.7  # g/cc
                    
                    # Adjust matrix density based on lithology if available
                    if 'LITHOLOGY' in st.session_state.well_data.columns:
                        matrix_density = np.where(
                            st.session_state.well_data['LITHOLOGY'] == 2, 2.71,  # Limestone
                            np.where(st.session_state.well_data['LITHOLOGY'] == 3, 2.87,  # Dolomite
                                    matrix_density))
                    
                    # Calculate porosity using density equation
                    phi = (matrix_density - st.session_state.well_data[porosity_col]) / (matrix_density - fluid_density)
                
                elif porosity_type == "Neutron":
                    # Convert neutron porosity from percentage to fraction
                    phi = st.session_state.well_data[porosity_col] / 100
                    
                    # Adjust for gas effect if needed
                    if fluid_type == "Gas":
                        phi = phi * 1.15  # Empirical correction for gas
                
                elif porosity_type == "Sonic":
                    # Default matrix transit time for sandstone (μs/ft)
                    matrix_time = 55.5  
                    
                    # Adjust fluid transit time based on fluid type
                    fluid_time = 189 if fluid_type == "Oil" else 220  # μs/ft
                    
                    # Calculate porosity using Wyllie time-average equation
                    phi = (st.session_state.well_data[porosity_col] - matrix_time) / (fluid_time - matrix_time)
                
                # Store results with proper clipping
                st.session_state.well_data['PHI_TOTAL'] = phi.clip(0, 0.4)
                st.session_state.calculations['PHI_calculated'] = True
                
                # Calculate effective porosity - if VSH not available, use total porosity
                if 'VSH' in st.session_state.well_data.columns:
                    st.session_state.well_data['PHI_EFF'] = st.session_state.well_data['PHI_TOTAL'] * (1 - st.session_state.well_data['VSH'])
                else:
                    st.session_state.well_data['PHI_EFF'] = st.session_state.well_data['PHI_TOTAL']
                    st.warning("VSH not calculated - using total porosity as effective porosity")
                st.session_state.calculations['PHI_EFF_calculated'] = True
                
                # Plot results
                fig = plt.figure(figsize=(8, 10), facecolor=st.session_state.plot_colors['background'])
                ax = fig.add_subplot(111)
                
                ax.plot(st.session_state.well_data['PHI_TOTAL'], st.session_state.well_data[st.session_state.curve_names[0]], 
                       color=st.session_state.plot_colors.get('PHI', 'blue'), linewidth=0.5, label='Total Porosity')
                
                ax.plot(st.session_state.well_data['PHI_EFF'], st.session_state.well_data[st.session_state.curve_names[0]], 
                       color=st.session_state.plot_colors.get('PHI_EFF', 'green'), linewidth=0.5, label='Effective Porosity')
                
                ax.set_xlabel('Porosity (v/v)')
                ax.set_ylabel(f"Depth ({get_units(st.session_state.curve_names[0])})")
                ax.set_title(f"Porosity ({porosity_type} Method, {fluid_type})")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
                ax.invert_yaxis()
                
                st.pyplot(fig)
                st.success(f"Porosity calculated using {porosity_type} method for {fluid_type}")
                
            except Exception as e:
                st.error(f"Error calculating porosity: {str(e)}")
                st.write("**Troubleshooting Tips:**")
                st.write("- Verify the selected log contains valid values")
                st.write("- Check for null/missing values in the log")
                st.write("- Calculate Vshale first for more accurate effective porosity")
    with tab3:  # Water Saturation Tab
      st.subheader("Water Saturation (Sw) with Gas/Shale Checks")
    
    # Check if PHI_TOTAL exists
    if 'PHI_TOTAL' not in st.session_state.well_data:
        st.error("⚠️ Calculate Total Porosity (PHI_TOTAL) first!")
        st.stop()

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        fluid_type = st.radio("Fluid Type", ["Oil", "Gas"], key='fluid_type_sw')
    with col2:
        shale_flag = st.checkbox("Shaly Formation?", help="Check if VSH > 10%")

    # Archie parameters (dynamic defaults)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Rw = st.number_input("Rw (ohm-m)", value=0.1, min_value=0.001, step=0.01, format="%.3f")
    with col2:
        a = st.slider("Tortuosity (a)", 0.5, 2.5, 1.0 if not shale_flag else 1.3, step=0.1)
    with col3:
        m = st.slider("Cementation (m)", 1.0, 3.0, 2.0 if not shale_flag else 2.5, step=0.1)
    with col4:
        n = st.slider("Saturation (n)", 1.0, 3.0, 2.0, step=0.1)

    # --- Gas Correction ---
    if fluid_type == "Gas":
        st.warning("Gas zone: Applying neutron-density correction (+10% porosity)")
        phi = st.session_state.well_data['PHI_TOTAL'] * 1.10  # Empirical correction
    else:
        phi = st.session_state.well_data['PHI_TOTAL']

    phi = phi.clip(0.01, 0.4)  # Avoid extremes
    rt = st.session_state.well_data['RT'].clip(0.1, 1000)  # Resistivity

    # --- Calculate Sw ---
    sw = ((a * Rw) / (phi ** m * rt)) ** (1/n)
    st.session_state.well_data['SW'] = np.clip(sw, 0, 1)

    # --- Plot Results ---
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(sw, st.session_state.well_data.DEPTH, 'b-', lw=0.5, label='Sw')
    
    # Highlight suspicious zones
    if shale_flag:
        ax.axvspan(0.7, 1.0, color='red', alpha=0.1, label='High Sw Risk (Shale)')
    if fluid_type == "Gas":
        ax.axvspan(0.0, 0.3, color='orange', alpha=0.1, label='Gas Effect Zone')
    
    ax.set_xlabel("Sw (v/v)")
    ax.set_ylabel("Depth (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    st.pyplot(fig)
    st.success(f"✅ Sw calculated ({fluid_type} zone, {'shaly' if shale_flag else 'clean'})")

    with tab4:  # Reservoir Flag Tab
     st.subheader("Reservoir Flagging with Custom Cutoffs")
    
    # Check prerequisites
    required_logs = ['PHI_TOTAL', 'SW']
    missing_logs = [log for log in required_logs if log not in st.session_state.well_data]
    if missing_logs:
        st.error(f"⚠️ Required logs missing: {', '.join(missing_logs)}")
        st.stop()

    # --- User Inputs for Cutoffs ---
    col1, col2, col3 = st.columns(3)
    with col1:
        phi_cutoff = st.slider(
            "Min Porosity (PHI_TOTAL)", 
            0.01, 0.40, 0.10, 0.01,
            help="Default: 0.10 for oil, 0.12 for gas"
        )
    with col2:
        sw_cutoff = st.slider(
            "Max Sw", 
            0.10, 1.0, 0.40, 0.05,  # Default Sw cutoff = 0.40
            help="Sw ≤ 0.40 for pay zones"
        )
    with col3:
        vsh_cutoff = st.slider(
            "Max Vsh", 
            0.0, 1.0, 0.60, 0.05,  # Default Vsh cutoff = 0.60
            help="Vsh ≤ 0.60 for clean zones"
        )

    # --- Flag Reservoir Zones ---
    if 'VSH' in st.session_state.well_data:
        st.session_state.well_data['RES_FLAG'] = (
            (st.session_state.well_data['PHI_TOTAL'] >= phi_cutoff) & 
            (st.session_state.well_data['SW'] <= sw_cutoff) &
            (st.session_state.well_data['VSH'] <= vsh_cutoff))
    else:
        st.warning("VSH log missing! Flagging without shale cutoff.")
        st.session_state.well_data['RES_FLAG'] = (
            (st.session_state.well_data['PHI_TOTAL'] >= phi_cutoff) & 
            (st.session_state.well_data['SW'] <= sw_cutoff))

    # --- Plot Results ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot curves
    ax.plot(st.session_state.well_data['PHI_TOTAL'], st.session_state.well_data.DEPTH, 'b-', label='Porosity')
    ax.plot(st.session_state.well_data['SW'], st.session_state.well_data.DEPTH, 'g-', label='Sw')
    
    # Highlight reservoir zones
    reservoir = st.session_state.well_data[st.session_state.well_data['RES_FLAG']]
    ax.fill_betweenx(reservoir.DEPTH, 0, 1, color='orange', alpha=0.3, label='Reservoir (Pay Zone)')
    
    # Add cutoff lines
    ax.axvline(phi_cutoff, color='blue', linestyle='--', lw=0.5, label='Porosity Cutoff')
    ax.axvline(sw_cutoff, color='green', linestyle='--', lw=0.5, label='Sw Cutoff')
    
    # Add Vsh cutoff line if available
    if 'VSH' in st.session_state.well_data:
        ax.plot(st.session_state.well_data['VSH'], st.session_state.well_data.DEPTH, 'r-', label='Vsh', alpha=0.5)
        ax.axvline(vsh_cutoff, color='red', linestyle='--', lw=0.5, label='Vsh Cutoff')

    ax.set_xlabel("Value (v/v)")
    ax.set_ylabel("Depth (m)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    st.pyplot(fig)

    with tab5:  # Net Pay Tab
     st.subheader("Net Pay Calculation")
    
    if 'RES_FLAG' not in st.session_state.well_data:
        st.error("⚠️ Flag reservoir zones first!")
        st.stop()
    
    # Calculate net pay
    reservoir = st.session_state.well_data[st.session_state.well_data['RES_FLAG']]
    if not reservoir.empty:
        net_pay_thickness = reservoir[st.session_state.curve_names[0]].max() - reservoir[st.session_state.curve_names[0]].min()
        
        # Calculate HCPV (Hydrocarbon Pore Volume)
        if 'VSH' in st.session_state.well_data:
            hcpv = (reservoir['PHI_EFF'] * (1 - reservoir['SW']) * (1 - reservoir['VSH'])).sum()
        else:
            hcpv = (reservoir['PHI_EFF'] * (1 - reservoir['SW'])).sum()
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Net Pay Thickness", f"{net_pay_thickness:.2f} {get_units(st.session_state.curve_names[0])}")
        with col2:
            st.metric("Hydrocarbon Pore Volume", f"{hcpv:.2f} m³/m")
        
        # Plot net pay
        fig = plt.figure(figsize=(8, 10), facecolor=st.session_state.plot_colors['background'])
        ax = fig.add_subplot(111)
        
        # Plot all data in background
        ax.plot(st.session_state.well_data['PHI_EFF'], 
               st.session_state.well_data[st.session_state.curve_names[0]], 
               color='lightgray', linewidth=0.5, alpha=0.5)
        
        # Highlight net pay zones
        ax.plot(reservoir['PHI_EFF'], reservoir[st.session_state.curve_names[0]], 
               color=st.session_state.plot_colors.get('reservoir', 'green'), 
               linewidth=1.0, label='Net Pay')
        
        ax.set_xlabel('Effective Porosity (v/v)')
        ax.set_ylabel(f"Depth ({get_units(st.session_state.curve_names[0])})")
        ax.set_title(f"Net Pay Zones (Thickness: {net_pay_thickness:.2f} {get_units(st.session_state.curve_names[0])})")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.invert_yaxis()
        
        st.pyplot(fig)
        
        # Export option
        if st.button("Export Net Pay Zones"):
            reservoir.to_csv("net_pay_zones.csv")
            st.success("Net pay zones exported to net_pay_zones.csv")
    else:
        st.warning("No reservoir zones found with current cutoffs")

# Statistics tab
elif selected_tab == "📈 Statistics":
    st.header("📈 Log Statistics")
    
    if st.button("Calculate Statistics"):
        stats = []
        for col in st.session_state.curve_names:
            data = st.session_state.well_data[col].dropna()
            stats.append({
                "Log": col,
                "Min": float(data.min()),
                "Max": float(data.max()),
                "Mean": float(data.mean()),
                "Std Dev": float(data.std()),
                "Count": len(data)
            })
        
        st.dataframe(pd.DataFrame(stats))
        st.success("Statistics calculated for all logs")

# Reservoir Analysis tab
elif selected_tab == "🛢️ Reservoir Analysis":
    st.header("🛢️ Reservoir Analysis")
    maintain_calculations()
    
    if not all([
        st.session_state.calculations['VSH_calculated'],
        st.session_state.calculations['PHI_calculated'],
        st.session_state.calculations['PHI_EFF_calculated'],
        st.session_state.calculations['SW_calculated']
    ]):
        st.warning("Complete all formation evaluations first (VSH, PHI, PHI_EFF, SW)")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            vsh_threshold = st.number_input("V_sh Threshold", value=0.3, key='vsh_threshold')
        with col2:
            sw_threshold = st.number_input("S_w Threshold", value=0.7, key='sw_threshold')
        with col3:
            phi_eff_threshold = st.number_input("Effective Porosity Threshold", value=0.1, key='phi_eff_threshold')
        
        depth_col = st.session_state.curve_names[0]
        col1, col2 = st.columns(2)
        with col1:
            min_depth = st.number_input("Minimum Depth", value=float(st.session_state.well_data[depth_col].min()), key='ra_min_depth')
        with col2:
            max_depth = st.number_input("Maximum Depth", value=float(st.session_state.well_data[depth_col].max()), key='ra_max_depth')
        
        if st.button("Run Reservoir Analysis"):
            try:
                # Filter data to depth range
                data = st.session_state.well_data[(st.session_state.well_data[depth_col] >= min_depth) & 
                                                (st.session_state.well_data[depth_col] <= max_depth)].copy()
                
                if data.empty:
                    st.warning("No data in selected depth range")
                else:
                    # Determine reservoir and pay flags
                    data['RESERVOIR_FLAG'] = np.where(
                        (data['VSH'] < vsh_threshold) & 
                        (data['SW'] < sw_threshold),
                        1, 0
                    )
                    
                    data['NET_PAY_FLAG'] = np.where(
                        (data['VSH'] < vsh_threshold) & 
                        (data['SW'] < sw_threshold) & 
                        (data['PHI_EFF'] >= phi_eff_threshold),
                        1, 0
                    )
                    
                    # Calculate net pay thickness and average properties
                    net_pay = data[data['NET_PAY_FLAG'] == 1]
                    if not net_pay.empty:
                        net_pay_thickness = net_pay[depth_col].max() - net_pay[depth_col].min()
                        avg_phi_eff = net_pay['PHI_EFF'].mean()
                        avg_sw = net_pay['SW'].mean()
                        avg_vsh = net_pay['VSH'].mean()
                    else:
                        net_pay_thickness = 0
                        avg_phi_eff = 0
                        avg_sw = 0
                        avg_vsh = 0
                    
                    # Plot results
                    fig = plt.figure(figsize=(14, 10), facecolor=st.session_state.plot_colors['background'])
                    gs = GridSpec(1, 5, width_ratios=[1, 1, 0.3, 0.3, 1])
                    
                    # Track 1: Vshale
                    ax1 = fig.add_subplot(gs[0])
                    ax1.plot(data['VSH'], data[depth_col], color=st.session_state.plot_colors.get('VSH', 'brown'), linewidth=0.5)
                    ax1.axvline(x=vsh_threshold, color='red', linestyle='--', linewidth=0.5)
                    ax1.set_xlabel('Vshale (v/v)')
                    ax1.set_ylabel(f"Depth ({get_units(depth_col)})")
                    ax1.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
                    ax1.invert_yaxis()
                    
                    # Track 2: Water Saturation
                    ax2 = fig.add_subplot(gs[1])
                    ax2.plot(data['SW'], data[depth_col], color=st.session_state.plot_colors.get('SW', 'darkblue'), linewidth=0.5)
                    ax2.axvline(x=sw_threshold, color='red', linestyle='--', linewidth=0.5)
                    ax2.set_xlabel('Sw (v/v)')
                    ax2.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
                    ax2.set_yticklabels([])
                    ax2.invert_yaxis()
                    
                    # Track 3: Reservoir Flag
                    ax3 = fig.add_subplot(gs[2])
                    ax3.set_xlim(0, 1)
                    ax3.fill_betweenx(data[depth_col], 0, data['RESERVOIR_FLAG'],
                                     color=st.session_state.plot_colors.get('reservoir', 'green'), alpha=0.5)
                    ax3.set_xticks([])
                    ax3.set_title('Reservoir')
                    ax3.invert_yaxis()
                    
                    # Track 4: Net Pay Flag
                    ax4 = fig.add_subplot(gs[3])
                    ax4.set_xlim(0, 1)
                    ax4.fill_betweenx(data[depth_col], 0, data['NET_PAY_FLAG'],
                                     color=st.session_state.plot_colors.get('reservoir', 'orange'), alpha=0.5)
                    ax4.set_xticks([])
                    ax4.set_title('Net Pay')
                    ax4.invert_yaxis()
                    
                    # Track 5: Effective Porosity
                    ax5 = fig.add_subplot(gs[4])
                    ax5.plot(data['PHI_EFF'], data[depth_col], color=st.session_state.plot_colors.get('PHI', 'purple'), linewidth=0.5)
                    ax5.axvline(x=phi_eff_threshold, color='red', linestyle='--', linewidth=0.5)
                    ax5.set_xlabel('Phi_eff (v/v)')
                    ax5.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
                    ax5.set_yticklabels([])
                    ax5.invert_yaxis()
                    
                    fig.suptitle(
                        f"Reservoir Analysis - Net Pay: {net_pay_thickness:.2f} {get_units(depth_col)}\n"
                        f"Thresholds: Vsh < {vsh_threshold}, Sw < {sw_threshold}, Phi_eff ≥ {phi_eff_threshold}"
                    )
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # Display reservoir summary
                    st.subheader("Reservoir Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Net Pay Thickness", f"{net_pay_thickness:.2f} {get_units(depth_col)}")
                    with col2:
                        st.metric("Avg Effective Porosity", f"{avg_phi_eff:.3f}")
                    with col3:
                        st.metric("Avg Water Saturation", f"{avg_sw:.3f}")
                    with col4:
                        st.metric("Avg Vshale", f"{avg_vsh:.3f}")
                    
                    st.success(
                        f"Reservoir analysis complete. Net pay thickness: {net_pay_thickness:.2f} {get_units(depth_col)}"
                    )
            
            except Exception as e:
                st.error(f"Failed to run reservoir analysis: {str(e)}")

# Quality Control tab
elif selected_tab == "✅ Quality Control":
    st.header("✅ Quality Control")
    
    method = st.selectbox("QC Method", [
        "Range Check", 
        "Spike Detection", 
        "Noise Detection", 
        "Repeat Section", 
        "Cross-Plot Validation",
        "Missing Data Check"
    ])
    
    if method == "Range Check":
        log = st.selectbox("Log", st.session_state.curve_names, key='range_log')
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.number_input("Minimum Value", value=-9999.0, key='range_min')
        with col2:
            max_val = st.number_input("Maximum Value", value=9999.0, key='range_max')
        
        if st.button("Run Range Check"):
            # Check for values outside range
            out_of_range = st.session_state.well_data[
                (st.session_state.well_data[log] < min_val) | 
                (st.session_state.well_data[log] > max_val)
            ]
            
            # Plot results
            fig, ax = plt.subplots(figsize=(8, 10), facecolor=st.session_state.plot_colors['background'])
            depth_col = st.session_state.curve_names[0]
            
            ax.plot(st.session_state.well_data[log], st.session_state.well_data[depth_col], 
                   color=st.session_state.plot_colors.get(log, 'blue'), linewidth=0.5, label='Log Data')
            ax.axvline(x=min_val, color='red', linestyle='--', linewidth=0.5, label='Min Threshold')
            ax.axvline(x=max_val, color='red', linestyle='--', linewidth=0.5, label='Max Threshold')
            
            if not out_of_range.empty:
                ax.scatter(out_of_range[log], out_of_range[depth_col], 
                          color=st.session_state.plot_colors.get('QC_fail', 'red'), s=20, label='Out of Range')
            
            ax.set_xlabel(f"{log} ({get_units(log)})")
            ax.set_ylabel(f"Depth ({get_units(depth_col)})")
            ax.set_title(f"Range Check for {log}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            ax.invert_yaxis()
            
            st.pyplot(fig)
            
            # Show results
            st.subheader("Results")
            st.write(f"Valid Range: {min_val} to {max_val}")
            st.write(f"Points Outside Range: {len(out_of_range)}")
            st.write(f"Percentage Bad: {100*len(out_of_range)/len(st.session_state.well_data):.2f}%")
            
            if not out_of_range.empty:
                st.write("Out of Range Points:")
                st.dataframe(out_of_range[[depth_col, log]])
    
    elif method == "Spike Detection":
        log = st.selectbox("Log", st.session_state.curve_names, key='spike_log')
        threshold = st.number_input("Threshold (Std Dev)", value=3.0, key='spike_threshold')
        
        if st.button("Run Spike Detection"):
            # Calculate differences
            diff = st.session_state.well_data[log].diff().abs()
            spikes = st.session_state.well_data[diff > threshold * st.session_state.well_data[log].std()]
            
            # Plot results
            fig, ax = plt.subplots(figsize=(8, 10), facecolor=st.session_state.plot_colors['background'])
            depth_col = st.session_state.curve_names[0]
            
            ax.plot(st.session_state.well_data[log], st.session_state.well_data[depth_col], 
                   color=st.session_state.plot_colors.get(log, 'blue'), linewidth=0.5, label='Log Data')
            
            if not spikes.empty:
                ax.scatter(spikes[log], spikes[depth_col], 
                          color=st.session_state.plot_colors.get('QC_fail', 'red'), s=20, label='Spikes')
            
            ax.set_xlabel(f"{log} ({get_units(log)})")
            ax.set_ylabel(f"Depth ({get_units(depth_col)})")
            ax.set_title(f"Spike Detection for {log} (Threshold: {threshold}σ)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            ax.invert_yaxis()
            
            st.pyplot(fig)
            
            # Show results
            st.subheader("Results")
            st.write(f"Threshold: {threshold} standard deviations")
            st.write(f"Spikes Detected: {len(spikes)}")
            st.write(f"Percentage Spikes: {100*len(spikes)/len(st.session_state.well_data):.2f}%")
            
            if not spikes.empty:
                st.write("Spike Points:")
                st.dataframe(spikes[[depth_col, log]])
    
    elif method == "Noise Detection":
        log = st.selectbox("Log", st.session_state.curve_names, key='noise_log')
        window_size = st.number_input("Window Size", value=5, min_value=1, max_value=100, key='noise_window')
        
        if st.button("Run Noise Detection"):
            # Calculate rolling statistics
            rolling_mean = st.session_state.well_data[log].rolling(window=window_size, center=True).mean()
            rolling_std = st.session_state.well_data[log].rolling(window=window_size, center=True).std()
            
            # Identify noisy points
            noise = st.session_state.well_data[
                (st.session_state.well_data[log] - rolling_mean).abs() > 2 * rolling_std
            ]
            
            # Plot results
            fig, ax = plt.subplots(figsize=(8, 10), facecolor=st.session_state.plot_colors['background'])
            depth_col = st.session_state.curve_names[0]
            
            ax.plot(st.session_state.well_data[log], st.session_state.well_data[depth_col], 
                   color=st.session_state.plot_colors.get(log, 'blue'), linewidth=0.5, label='Log Data')
            ax.plot(rolling_mean, st.session_state.well_data[depth_col], 
                   color=st.session_state.plot_colors.get('QC_pass', 'green'), linewidth=1, label='Rolling Mean')
            
            if not noise.empty:
                ax.scatter(noise[log], noise[depth_col], 
                          color=st.session_state.plot_colors.get('QC_fail', 'red'), s=20, label='Noisy Points')
            
            ax.set_xlabel(f"{log} ({get_units(log)})")
            ax.set_ylabel(f"Depth ({get_units(depth_col)})")
            ax.set_title(f"Noise Detection for {log} (Window: {window_size} samples)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            ax.invert_yaxis()
            
            st.pyplot(fig)
            
            # Show results
            st.subheader("Results")
            st.write(f"Window Size: {window_size} samples")
            st.write(f"Noisy Points: {len(noise)}")
            st.write(f"Percentage Noisy: {100*len(noise)/len(st.session_state.well_data):.2f}%")
            
            if not noise.empty:
                st.write("Noisy Points:")
                st.dataframe(noise[[depth_col, log]])
    
    elif method == "Repeat Section":
        log = st.selectbox("Log", st.session_state.curve_names, key='repeat_log')
        depth_col = st.session_state.curve_names[0]
        col1, col2 = st.columns(2)
        with col1:
            top = st.number_input("Top Depth", value=float(st.session_state.well_data[depth_col].min()), key='repeat_top')
        with col2:
            base = st.number_input("Base Depth", value=float(st.session_state.well_data[depth_col].max()), key='repeat_base')
        
        if st.button("Run Repeat Section Analysis"):
            # Get repeat section and main section
            repeat_section = st.session_state.well_data[
                (st.session_state.well_data[depth_col] >= top) & 
                (st.session_state.well_data[depth_col] <= base)
            ]
            main_section = st.session_state.well_data[
                (st.session_state.well_data[depth_col] < top) | 
                (st.session_state.well_data[depth_col] > base)
            ]
            
            if len(repeat_section) == 0:
                st.warning("No data in repeat section")
            else:
                # Plot results
                fig, ax = plt.subplots(figsize=(8, 10), facecolor=st.session_state.plot_colors['background'])
                
                ax.plot(main_section[log], main_section[depth_col], 
                       color=st.session_state.plot_colors.get(log, 'blue'), linewidth=0.5, label='Main Log')
                ax.plot(repeat_section[log], repeat_section[depth_col], 
                       color=st.session_state.plot_colors.get('QC_warning', 'orange'), linewidth=0.5, label='Repeat Section')
                
                ax.set_xlabel(f"{log} ({get_units(log)})")
                ax.set_ylabel(f"Depth ({get_units(depth_col)})")
                ax.set_title(f"Repeat Section Analysis for {log}")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
                ax.invert_yaxis()
                
                st.pyplot(fig)
                
                # Calculate statistics
                repeat_mean = repeat_section[log].mean()
                main_mean = main_section[log].mean()
                repeat_std = repeat_section[log].std()
                main_std = main_section[log].std()
                
                # Show results
                st.subheader("Results")
                st.write(f"Repeat Section: {top} to {base} {get_units(depth_col)}")
                st.write(f"Repeat Section Mean: {repeat_mean:.4f}")
                st.write(f"Main Log Mean: {main_mean:.4f}")
                st.write(f"Difference: {abs(repeat_mean - main_mean):.4f}")
                st.write(f"Repeat Section Std Dev: {repeat_std:.4f}")
                st.write(f"Main Log Std Dev: {main_std:.4f}")
                st.write(f"Percentage Difference: {100*abs(repeat_mean - main_mean)/main_mean:.2f}%")
    
    elif method == "Cross-Plot Validation":
        col1, col2 = st.columns(2)
        with col1:
            x_log = st.selectbox("X Log", st.session_state.curve_names, key='cross_x')
        with col2:
            y_log = st.selectbox("Y Log", st.session_state.curve_names, key='cross_y')
        
        tolerance = st.number_input("Tolerance (%)", value=5.0, key='cross_tolerance') / 100  # Convert to fraction
        
        if st.button("Run Cross-Plot Validation"):
            # Calculate expected relationship (linear regression)
            valid_data = st.session_state.well_data[[x_log, y_log]].dropna()
            if len(valid_data) < 2:
                st.warning("Not enough data for cross-plot validation")
            else:
                # Fit linear model
                coeff = np.polyfit(valid_data[x_log], valid_data[y_log], 1)
                predicted = np.polyval(coeff, st.session_state.well_data[x_log])
                
                # Calculate residuals
                residuals = st.session_state.well_data[y_log] - predicted
                relative_error = np.abs(residuals / predicted)
                
                # Identify outliers
                outliers = st.session_state.well_data[relative_error > tolerance]
                
                # Plot crossplot
                fig = plt.figure(figsize=(12, 6), facecolor=st.session_state.plot_colors['background'])
                ax1 = fig.add_subplot(121)
                ax1.scatter(valid_data[x_log], valid_data[y_log], alpha=0.5, 
                           color=st.session_state.plot_colors.get('QC_pass', 'green'), label='Valid Data')
                
                if not outliers.empty:
                    ax1.scatter(outliers[x_log], outliers[y_log], 
                              color=st.session_state.plot_colors.get('QC_fail', 'red'), label='Outliers')
                
                # Plot regression line
                x_vals = np.linspace(valid_data[x_log].min(), valid_data[x_log].max(), 100)
                y_vals = np.polyval(coeff, x_vals)
                ax1.plot(x_vals, y_vals, '--', color='black', label='Regression Line')
                
                ax1.set_xlabel(f"{x_log} ({get_units(x_log)})")
                ax1.set_ylabel(f"{y_log} ({get_units(y_log)})")
                ax1.set_title(f"{y_log} vs {x_log} Crossplot")
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
                
                # Plot depth track with outliers
                depth_col = st.session_state.curve_names[0]
                ax2 = fig.add_subplot(122)
                ax2.plot(st.session_state.well_data[y_log], st.session_state.well_data[depth_col], 
                        color=st.session_state.plot_colors.get(y_log, 'blue'), linewidth=0.5, label=y_log)
                
                if not outliers.empty:
                    ax2.scatter(outliers[y_log], outliers[depth_col], 
                              color=st.session_state.plot_colors.get('QC_fail', 'red'), s=20, label='Outliers')
                
                ax2.set_xlabel(f"{y_log} ({get_units(y_log)})")
                ax2.set_ylabel(f"Depth ({get_units(depth_col)})")
                ax2.set_title("Outliers in Depth")
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
                ax2.invert_yaxis()
                
                fig.tight_layout()
                st.pyplot(fig)
                
                # Show results
                st.subheader("Results")
                st.write(f"X Log: {x_log}, Y Log: {y_log}")
                st.write(f"Tolerance: {tolerance*100:.1f}%")
                st.write(f"Regression Equation: y = {coeff[0]:.4f}x + {coeff[1]:.4f}")
                st.write(f"Outliers Detected: {len(outliers)}")
                st.write(f"Percentage Outliers: {100*len(outliers)/len(st.session_state.well_data):.2f}%")
                
                if not outliers.empty:
                    st.write("Outlier Points:")
                    st.dataframe(outliers[[depth_col, x_log, y_log]])
    
    elif method == "Missing Data Check":
        log = st.selectbox("Log", st.session_state.curve_names, key='missing_log')
        
        if st.button("Run Missing Data Check"):
            # Identify missing data
            missing = st.session_state.well_data[st.session_state.well_data[log].isna()]
            
            # Plot results
            fig, ax = plt.subplots(figsize=(8, 10), facecolor=st.session_state.plot_colors['background'])
            depth_col = st.session_state.curve_names[0]
            
            ax.plot(st.session_state.well_data[log], st.session_state.well_data[depth_col], 
                   color=st.session_state.plot_colors.get(log, 'blue'), linewidth=0.5, label='Log Data')
            
            if not missing.empty:
                for _, row in missing.iterrows():
                    ax.axhline(y=row[depth_col], color=st.session_state.plot_colors.get('QC_fail', 'red'), 
                              linewidth=0.5, alpha=0.3)
            
            ax.set_xlabel(f"{log} ({get_units(log)})")
            ax.set_ylabel(f"Depth ({get_units(depth_col)})")
            ax.set_title(f"Missing Data Check for {log}")
            ax.grid(True, linestyle='--', alpha=0.5, color=st.session_state.plot_colors['grid'])
            ax.invert_yaxis()
            
            st.pyplot(fig)
            
            # Show results
            st.subheader("Results")
            st.write(f"Total Samples: {len(st.session_state.well_data)}")
            st.write(f"Missing Samples: {len(missing)}")
            st.write(f"Percentage Missing: {100*len(missing)/len(st.session_state.well_data):.2f}%")
            
            if not missing.empty:
                st.write("Missing Data Depths:")
                st.dataframe(missing[[depth_col]])

# Settings tab
elif selected_tab == "⚙️ Settings":
    st.header("⚙️ Plot Settings")
    
    st.subheader("Plot Color Settings")
    
    # Create color pickers for each plot element
    for name, default_color in st.session_state.plot_colors.items():
        st.session_state.plot_colors[name] = st.color_picker(
            f"{name.replace('_', ' ').title()} Color",
            value=default_color,
            key=f"color_{name}"
        )
    
    if st.button("Apply Colors"):
        st.success("Color settings applied")