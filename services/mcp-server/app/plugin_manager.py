import os
import sys
import yaml
import importlib
import importlib.util
import re
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import traceback

logger = logging.getLogger(__name__)

def resolve_env_vars(config):
    """Recursively resolve environment variables in configuration"""
    if isinstance(config, dict):
        return {key: resolve_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [resolve_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Replace ${VAR_NAME} with environment variable value
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Return original if env var not found
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, config)
    else:
        return config

class Plugin:
    """Base class for plugins"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unnamed")
        self.version = config.get("version", "1.0.0")
        self.tools = []
    
    def get_tools(self) -> list:
        """Get tools provided by this plugin"""
        # Default implementation returns empty list
        # Concrete plugin classes should override this method
        logger.info(f"Plugin '{self.name}' provides {len(self.tools)} tools")
        return self.tools
    
    def initialize(self):
        """Initialize the plugin"""
        logger.info(f"Initializing plugin '{self.name}' v{self.version}")
        pass
    
    def cleanup(self):
        """Cleanup plugin resources"""
        logger.info(f"Cleaning up plugin '{self.name}'")
        pass

class PluginManager:
    """Manages plugin loading and lifecycle"""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_dir = os.getenv("PLUGIN_DIR", "/app/plugins")
    
    def load_plugins(self, plugin_dir: Optional[str] = None):
        """Load all plugins from directory"""
        if plugin_dir:
            self.plugin_dir = plugin_dir
        
        plugin_path = Path(self.plugin_dir)
        if not plugin_path.exists():
            logger.warning(f"Plugin directory {plugin_path} does not exist")
            return
        
        # Add plugin directory to Python path
        if str(plugin_path) not in sys.path:
            sys.path.insert(0, str(plugin_path))
        
        # Load each plugin
        for plugin_folder in plugin_path.iterdir():
            if plugin_folder.is_dir() and not plugin_folder.name.startswith("_"):
                try:
                    self._load_plugin(plugin_folder)
                except Exception as e:
                    logger.error(f"Failed to load plugin from {plugin_folder}: {e}")
                    logger.error(traceback.format_exc())
    
    def _load_plugin(self, plugin_folder: Path):
        """Load a single plugin"""
        # Load plugin config
        config_file = plugin_folder / "plugin.yaml"
        if not config_file.exists():
            logger.warning(f"No plugin.yaml found in {plugin_folder}")
            return
        
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        # Resolve environment variables in configuration
        config = resolve_env_vars(config)
        
        plugin_name = config.get("name", plugin_folder.name)
        
        # Load plugin module
        module_name = f"plugins.{plugin_folder.name}"
        init_file = plugin_folder / "__init__.py"
        
        if init_file.exists():
            spec = importlib.util.spec_from_file_location(module_name, init_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get plugin class
            plugin_class_name = config.get("class", "Plugin")
            if hasattr(module, plugin_class_name):
                plugin_class = getattr(module, plugin_class_name)
                plugin_instance = plugin_class(config)
                plugin_instance.initialize()
                
                self.plugins[plugin_name] = plugin_instance
                logger.info(f"Loaded plugin '{plugin_name}' v{plugin_instance.version}")
            else:
                logger.error(f"Plugin class '{plugin_class_name}' not found in {module_name}")
    
    def reload_plugins(self):
        """Reload all plugins (hot reload)"""
        # Cleanup existing plugins
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin: {e}")
        
        self.plugins.clear()
        
        # Reload plugins
        self.load_plugins()
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        return {
            plugin_name: {
                "version": plugin.version,
                "loaded": True
            }
            for plugin_name, plugin in self.plugins.items()
        }