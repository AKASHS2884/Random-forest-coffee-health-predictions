from django.apps import AppConfig


class ApiConfig(AppConfig):
    """
    Configuration class for the API application.
    
    This class defines application-level settings and initialization logic.
    Django loads this configuration when the app is included in INSTALLED_APPS.
    """
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    verbose_name = 'Coffee Health Prediction API'
    
    def ready(self):
        """
        Called when Django starts and the app registry is populated.
        
        Use this method to:
        - Register signal handlers
        - Perform startup checks
        - Initialize caches or connections
        - Import and register custom components
        
        This method runs once per application lifecycle.
        """
        pass
