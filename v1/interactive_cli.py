"""
Interactive CLI for Manifold System v1
A menu-driven interface for exploring manifold projections
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from manifold_system import ManifoldSystem
from manifold_learning import ManifoldLearner
from manifold_viz import ManifoldVisualizer


class InteractiveManifold:
    def __init__(self):
        self.system = ManifoldSystem()
        self.learner = ManifoldLearner()
        self.viz = ManifoldVisualizer()
        self.data = None
        self.X = None
        self.entity_ids = None
        self.feature_names = None
        self.projections = {}
        self.labels = None
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title):
        """Print section header"""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70 + "\n")
    
    def print_menu(self, title, options):
        """Print menu options"""
        self.print_header(title)
        for key, desc in options.items():
            print(f"  [{key}] {desc}")
        print()
    
    def get_choice(self, valid_choices):
        """Get user choice with validation"""
        while True:
            choice = input("Choose an option: ").strip().lower()
            if choice in valid_choices:
                return choice
            print(f"Invalid choice. Please choose from: {', '.join(valid_choices)}")
    
    def load_data_menu(self):
        """Interactive data loading"""
        self.print_header("Load Data")
        
        print("Options:")
        print("  [1] Load CSV file")
        print("  [2] Load example synthetic data")
        print("  [3] Use PSID/economic dataset (if available)")
        print()
        
        choice = self.get_choice(['1', '2', '3'])
        
        if choice == '1':
            return self.load_csv_interactive()
        elif choice == '2':
            return self.load_synthetic_data()
        elif choice == '3':
            return self.load_economic_data()
    
    def load_csv_interactive(self):
        """Load CSV with interactive column selection"""
        while True:
            filepath = input("\nEnter CSV file path (or 'back'): ").strip()
            
            if filepath.lower() == 'back':
                return False
            
            if not os.path.exists(filepath):
                print(f"❌ File not found: {filepath}")
                continue
            
            try:
                print("Loading file...")
                df = pd.read_csv(filepath)
                print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
                print(f"\nColumns: {list(df.columns)}")
                
                # Interactive column selection
                print("\n" + "-" * 70)
                entity_col = input("Enter entity ID column name: ").strip()
                if entity_col not in df.columns:
                    print(f"❌ Column '{entity_col}' not found")
                    continue
                
                has_time = input("Does data have timestamps? (y/n): ").strip().lower() == 'y'
                time_col = None
                if has_time:
                    time_col = input("Enter timestamp column name: ").strip()
                    if time_col not in df.columns:
                        print(f"❌ Column '{time_col}' not found")
                        continue
                
                # Feature selection
                print("\nFeature selection:")
                print("  [1] Use all numeric columns")
                print("  [2] Specify feature columns")
                print("  [3] Select by pattern (e.g., 'feature_')")
                
                feat_choice = self.get_choice(['1', '2', '3'])
                
                if feat_choice == '1':
                    exclude = [entity_col]
                    if time_col:
                        exclude.append(time_col)
                    feature_cols = df.select_dtypes(include=[np.number]).columns
                    feature_cols = [c for c in feature_cols if c not in exclude]
                
                elif feat_choice == '2':
                    feat_input = input("Enter feature column names (comma-separated): ")
                    feature_cols = [c.strip() for c in feat_input.split(',')]
                
                elif feat_choice == '3':
                    pattern = input("Enter pattern: ").strip()
                    feature_cols = [c for c in df.columns if pattern in c]
                
                print(f"\n✓ Selected {len(feature_cols)} features")
                print(f"  First 5: {feature_cols[:5]}")
                
                # Optional: label column for coloring
                has_label = input("\nDo you have a label/cluster column? (y/n): ").strip().lower() == 'y'
                label_col = None
                if has_label:
                    label_col = input("Enter label column name: ").strip()
                    if label_col in df.columns:
                        self.labels = df.groupby(entity_col)[label_col].first().values
                
                # Ingest
                print("\nIngesting data...")
                self.system.ingest_dataframe(
                    df,
                    entity_id_col=entity_col,
                    timestamp_col=time_col,
                    feature_cols=feature_cols,
                    preprocess=True
                )
                
                self.data = df
                print("✓ Data loaded successfully!")
                
                input("\nPress Enter to continue...")
                return True
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                input("\nPress Enter to continue...")
                return False
    
    def load_synthetic_data(self):
        """Generate synthetic data"""
        print("\n" + "-" * 70)
        print("Generating synthetic data...")
        
        try:
            n_entities = int(input("Number of entities (default 200): ").strip() or "200")
            n_features = int(input("Number of features (default 30): ").strip() or "30")
            n_timepoints = int(input("Number of timepoints (default 1): ").strip() or "1")
            
            # Generate data
            from example_usage import generate_synthetic_data
            df = generate_synthetic_data(n_entities, n_features, n_timepoints, add_clusters=True)
            
            feature_cols = [c for c in df.columns if c.startswith('feature_')]
            
            self.system.ingest_dataframe(
                df,
                entity_id_col='entity_id',
                timestamp_col='timestamp',
                feature_cols=feature_cols,
                metadata_cols=['cluster']
            )
            
            self.data = df
            self.labels = df.groupby('entity_id')['cluster'].first().values
            
            print(f"✓ Generated {len(df)} rows with {len(feature_cols)} features")
            input("\nPress Enter to continue...")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter to continue...")
            return False
    
    def load_economic_data(self):
        """Prompt for economic dataset"""
        print("\n" + "-" * 70)
        print("Economic Dataset Loader")
        print("\nTo use this, you need to download:")
        print("  - PSID: https://psidonline.isr.umich.edu/")
        print("  - Survey of Consumer Finances: https://www.federalreserve.gov/econres/scfindex.htm")
        print("\nAfter downloading, use option [1] to load the CSV file")
        input("\nPress Enter to continue...")
        return False
    
    def prepare_features(self):
        """Prepare feature matrix"""
        if self.data is None:
            print("❌ No data loaded!")
            input("\nPress Enter to continue...")
            return False
        
        self.print_header("Prepare Features")
        
        print("Preprocessing options:")
        print("  [1] No preprocessing")
        print("  [2] Standardize (z-score)")
        print("  [3] Normalize (0-1)")
        print()
        
        choice = self.get_choice(['1', '2', '3'])
        
        standardize = choice == '2'
        normalize = choice == '3'
        
        print("\nPreparing feature matrix...")
        self.X, self.entity_ids, self.feature_names = self.system.get_feature_matrix(
            standardize=standardize,
            normalize=normalize
        )
        
        print(f"✓ Feature matrix: {self.X.shape}")
        print(f"  Entities: {len(self.entity_ids)}")
        print(f"  Features: {len(self.feature_names)}")
        
        input("\nPress Enter to continue...")
        return True
    
    def run_projection_menu(self):
        """Interactive projection menu"""
        if self.X is None:
            print("❌ No features prepared! Run 'Prepare Features' first.")
            input("\nPress Enter to continue...")
            return
        
        self.print_header("Run Manifold Projection")
        
        # Check available methods
        deps = self.learner.check_dependencies()
        
        methods = {
            '1': ('PCA', 'pca'),
            '2': ('t-SNE', 'tsne'),
            '3': ('Isomap', 'isomap'),
            '4': ('LLE', 'lle'),
            '5': ('MDS', 'mds'),
        }
        
        if deps['umap']:
            methods['6'] = ('UMAP', 'umap')
        
        print("Available methods:")
        for key, (name, _) in methods.items():
            print(f"  [{key}] {name}")
        print()
        
        choice = self.get_choice(list(methods.keys()))
        method_name, method_code = methods[choice]
        
        # Get parameters
        print(f"\nParameters for {method_name}:")
        n_components = int(input("Number of components (2 or 3, default 2): ").strip() or "2")
        
        # Method-specific parameters
        kwargs = {}
        if method_code in ['tsne', 'umap', 'isomap', 'lle']:
            if method_code == 'tsne':
                perplexity = float(input("Perplexity (default 30): ").strip() or "30")
                kwargs['perplexity'] = perplexity
            elif method_code == 'umap':
                n_neighbors = int(input("Number of neighbors (default 15): ").strip() or "15")
                min_dist = float(input("Min distance (default 0.1): ").strip() or "0.1")
                kwargs['n_neighbors'] = n_neighbors
                kwargs['min_dist'] = min_dist
            else:
                n_neighbors = int(input("Number of neighbors (default 5): ").strip() or "5")
                kwargs['n_neighbors'] = n_neighbors
        
        # Run projection
        print(f"\nRunning {method_name}...")
        try:
            if method_code == 'pca':
                proj = self.learner.project_pca(self.X, n_components, entity_ids=self.entity_ids)
            elif method_code == 'tsne':
                proj = self.learner.project_tsne(self.X, n_components, entity_ids=self.entity_ids, **kwargs)
            elif method_code == 'isomap':
                proj = self.learner.project_isomap(self.X, n_components, entity_ids=self.entity_ids, **kwargs)
            elif method_code == 'lle':
                proj = self.learner.project_lle(self.X, n_components, entity_ids=self.entity_ids, **kwargs)
            elif method_code == 'mds':
                proj = self.learner.project_mds(self.X, n_components, entity_ids=self.entity_ids)
            elif method_code == 'umap':
                proj = self.learner.project_umap(self.X, n_components, entity_ids=self.entity_ids, **kwargs)
            
            self.projections[method_code] = proj
            
            print(f"✓ {method_name} complete!")
            if 'total_explained_variance' in proj.metadata:
                print(f"  Variance explained: {proj.metadata['total_explained_variance']:.2%}")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")
    
    def visualize_menu(self):
        """Interactive visualization menu"""
        if not self.projections:
            print("❌ No projections available! Run a projection first.")
            input("\nPress Enter to continue...")
            return
        
        self.print_header("Visualize Projections")
        
        print("Available projections:")
        proj_list = list(self.projections.items())
        for i, (name, proj) in enumerate(proj_list, 1):
            print(f"  [{i}] {proj.method} ({proj.target_dim}D)")
        print()
        
        choice = input("Choose projection to visualize (number): ").strip()
        try:
            idx = int(choice) - 1
            name, proj = proj_list[idx]
        except:
            print("Invalid choice")
            input("\nPress Enter to continue...")
            return
        
        # Visualization options
        print("\nVisualization options:")
        print("  [1] Simple scatter plot")
        print("  [2] Colored by labels (if available)")
        print("  [3] With entity ID labels")
        print("  [4] Save to file")
        print()
        
        vis_choice = self.get_choice(['1', '2', '3', '4'])
        
        try:
            import matplotlib.pyplot as plt
            
            if vis_choice in ['1', '2', '3']:
                show_labels = vis_choice == '3'
                use_colors = vis_choice == '2' and self.labels is not None
                
                labels = self.labels if use_colors else None
                
                if proj.target_dim == 2:
                    fig, ax = self.viz.plot_projection_2d(
                        proj,
                        labels=labels,
                        show_labels=show_labels
                    )
                elif proj.target_dim == 3:
                    fig, ax = self.viz.plot_projection_3d(
                        proj,
                        labels=labels
                    )
                
                plt.show()
            
            elif vis_choice == '4':
                filename = input("Enter filename (e.g., my_projection.png): ").strip()
                if not filename:
                    filename = f"projection_{name}.png"
                
                show_labels = input("Show entity labels? (y/n): ").strip().lower() == 'y'
                use_colors = input("Color by labels? (y/n): ").strip().lower() == 'y' and self.labels is not None
                
                labels = self.labels if use_colors else None
                
                if proj.target_dim == 2:
                    fig, ax = self.viz.plot_projection_2d(
                        proj,
                        labels=labels,
                        show_labels=show_labels
                    )
                else:
                    fig, ax = self.viz.plot_projection_3d(
                        proj,
                        labels=labels
                    )
                
                self.viz.save_figure(fig, filename)
                print(f"✓ Saved to {filename}")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")
    
    def compare_methods_menu(self):
        """Compare multiple methods"""
        if self.X is None:
            print("❌ No features prepared!")
            input("\nPress Enter to continue...")
            return
        
        self.print_header("Compare Methods")
        
        print("This will run multiple methods and compare them.")
        print("Methods: PCA, Isomap, t-SNE" + (", UMAP" if self.learner.check_dependencies()['umap'] else ""))
        print()
        
        proceed = input("Continue? (y/n): ").strip().lower() == 'y'
        if not proceed:
            return
        
        methods = ['pca', 'isomap', 'tsne']
        if self.learner.check_dependencies()['umap']:
            methods.append('umap')
        
        print("\nRunning comparisons (this may take a minute)...")
        try:
            projections = self.learner.compare_methods(
                self.X,
                methods=methods,
                n_components=2,
                entity_ids=self.entity_ids
            )
            
            # Update stored projections
            for name, proj in projections.items():
                self.projections[name] = proj
            
            # Visualize comparison
            import matplotlib.pyplot as plt
            fig, axes = self.viz.compare_projections(projections, labels=self.labels)
            plt.show()
            
            print("✓ Comparison complete!")
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")
    
    def trajectory_menu(self):
        """Trajectory analysis"""
        print("\n⚠ Trajectory analysis requires temporal data")
        print("This feature analyzes how entities move through manifold space over time.")
        print("\nNot yet implemented in interactive mode.")
        input("\nPress Enter to continue...")
    
    def export_menu(self):
        """Export results"""
        if not self.projections:
            print("❌ No projections to export!")
            input("\nPress Enter to continue...")
            return
        
        self.print_header("Export Results")
        
        print("Export options:")
        print("  [1] Export projection to CSV")
        print("  [2] Export all projections")
        print()
        
        choice = self.get_choice(['1', '2'])
        
        try:
            if choice == '1':
                # Select projection
                proj_list = list(self.projections.items())
                print("\nAvailable projections:")
                for i, (name, proj) in enumerate(proj_list, 1):
                    print(f"  [{i}] {proj.method}")
                
                idx = int(input("\nChoose projection: ").strip()) - 1
                name, proj = proj_list[idx]
                
                # Export
                filename = input(f"Filename (default: projection_{name}.csv): ").strip()
                if not filename:
                    filename = f"projection_{name}.csv"
                
                df = pd.DataFrame(
                    proj.embedding,
                    columns=[f'component_{i+1}' for i in range(proj.target_dim)]
                )
                df.insert(0, 'entity_id', proj.entity_ids)
                df.to_csv(filename, index=False)
                
                print(f"✓ Exported to {filename}")
            
            elif choice == '2':
                for name, proj in self.projections.items():
                    filename = f"projection_{name}.csv"
                    df = pd.DataFrame(
                        proj.embedding,
                        columns=[f'component_{i+1}' for i in range(proj.target_dim)]
                    )
                    df.insert(0, 'entity_id', proj.entity_ids)
                    df.to_csv(filename, index=False)
                    print(f"✓ Exported {name} to {filename}")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter to continue...")
    
    def main_menu(self):
        """Main menu loop"""
        while True:
            self.clear_screen()
            
            print("\n" + "╔" + "═" * 68 + "╗")
            print("║" + " " * 18 + "MANIFOLD SYSTEM v1 - INTERACTIVE" + " " * 18 + "║")
            print("╚" + "═" * 68 + "╝\n")
            
            # Show status
            status_items = []
            if self.data is not None:
                status_items.append(f"Data: {len(self.data)} rows")
            if self.X is not None:
                status_items.append(f"Features: {self.X.shape}")
            if self.projections:
                status_items.append(f"Projections: {len(self.projections)}")
            
            if status_items:
                print("Status: " + " | ".join(status_items))
                print()
            
            # Menu
            print("Main Menu:")
            print("  [1] Load Data")
            print("  [2] Prepare Features")
            print("  [3] Run Projection")
            print("  [4] Visualize")
            print("  [5] Compare Methods")
            print("  [6] Trajectory Analysis")
            print("  [7] Export Results")
            print("  [q] Quit")
            print()
            
            choice = input("Choose an option: ").strip().lower()
            
            if choice == '1':
                self.load_data_menu()
            elif choice == '2':
                self.prepare_features()
            elif choice == '3':
                self.run_projection_menu()
            elif choice == '4':
                self.visualize_menu()
            elif choice == '5':
                self.compare_methods_menu()
            elif choice == '6':
                self.trajectory_menu()
            elif choice == '7':
                self.export_menu()
            elif choice == 'q':
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice")
                input("\nPress Enter to continue...")


if __name__ == "__main__":
    app = InteractiveManifold()
    app.main_menu()
