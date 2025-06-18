import pandas as pd

from quantfin.data.data_manager        import load_market_snapshot
from quantfin.workflows.daily_workflow import DailyWorkflow
from quantfin.calibration.iv_surface   import VolatilitySurface
from quantfin.plotting.smile_plotter   import plot_smiles_by_expiry, plot_iv_surface
from quantfin.atoms.stock              import Stock
from quantfin.atoms.rate               import Rate
from quantfin.techniques.fft           import FFTTechnique
from quantfin.calibration.calibrator   import Calibrator
from quantfin.models.bsm               import BSMModel
from quantfin.models.merton_jump       import MertonJumpModel
from quantfin.workflows.configs.bsm_config    import BSM_WORKFLOW_CONFIG
from quantfin.workflows.configs.merton_config import MERTON_WORKFLOW_CONFIG

def main(ticker: str, snapshot_date: str):
    """
    Main workflow to calibrate models and generate visualization plots.
    """
    # Load Data
    market_data = load_market_snapshot(ticker, snapshot_date)
    if market_data is None: return
    
    stock = Stock(spot=market_data['spot_price'].iloc[0]) # Rate/div will be fit inside workflow
    
    # Calibrate Models
    models_to_run = [BSM_WORKFLOW_CONFIG, MERTON_WORKFLOW_CONFIG] # Add more here
    calibrated_models = {}
    
    for model_config in models_to_run:
        workflow = DailyWorkflow(market_data=market_data, model_config=model_config)
        workflow.run()
        model_name = workflow.results['Model']
        calibrated_params = workflow.results['Calibrated Params']
        calibrated_models[model_name] = model_config['model_class'](params=calibrated_params)

    # Generate IV Surfaces
    iv_solver = FFTTechnique() 
    
    # Generate Market IV Surface
    market_surface_generator = VolatilitySurface(market_data)
    market_surface = market_surface_generator.calculate_market_iv(stock, Rate(rate=0.05), iv_solver).surface
    
    # Generate Model IV Surfaces
    model_surfaces = {}
    for name, model in calibrated_models.items():
        model_surface_generator = VolatilitySurface(market_data)
        # Use the technique from the calibrator for consistency
        technique = Calibrator(model, market_data, stock, Rate(rate=0.05)).technique
        model_surfaces[name] = model_surface_generator.calculate_model_iv(stock, Rate(rate=0.05), model, technique, iv_solver).surface

    # 4. Create Plots ---
    print("\nGenerating visualization plots...")
    
    # 2D Smile Plots
    plot_smiles_by_expiry(market_surface, model_surfaces, ticker, snapshot_date)
    
    # 3D Surface Plot
    plot_iv_surface(market_surface, model_surfaces, ticker)
