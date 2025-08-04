import os
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import logging
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class PluginReloadHandler(FileSystemEventHandler):
    """Handles file system events for hot reload"""
    
    def __init__(self, plugin_loader):
        self.plugin_loader = plugin_loader
        self.last_reload = 0
        
    def on_modified(self, event):
        if not event.is_directory and (event.src_path.endswith('.py') or event.src_path.endswith('.yaml')):
            current_time = time.time()
            # Debounce - only reload if 2 seconds have passed
            if current_time - self.last_reload > 2:
                logger.info(f"Detected change in {event.src_path}, reloading plugins...")
                self.plugin_loader.reload_plugins()
                self.last_reload = current_time

class PluginLoader:
    """Dynamic plugin loader with hot reload support"""
    
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        self.plugin_dir = "/app/plugins"
        self.observer = None
        self._lock = threading.Lock()
        
    def load_plugins(self, plugin_dir: Optional[str] = None):
        """Load all plugins from directory"""
        if plugin_dir:
            self.plugin_dir = plugin_dir
            
        with self._lock:
            # Clear existing plugins
            self.plugins.clear()
            
            # Ensure plugin directory is in Python path
            plugin_path = Path(self.plugin_dir)
            if str(plugin_path) not in sys.path:
                sys.path.insert(0, str(plugin_path))
            
            if not plugin_path.exists():
                logger.warning(f"Plugin directory {plugin_path} does not exist")
                return
            
            # Load each plugin directory
            for item in plugin_path.iterdir():
                if item.is_dir() and not item.name.startswith('_'):
                    try:
                        self._load_plugin(item)
                    except Exception as e:
                        logger.error(f"Failed to load plugin {item.name}: {e}", exc_info=True)
        
        # Setup hot reload in development
        if os.getenv("ENVIRONMENT", "development") == "development":
            self._setup_hot_reload()
    
    def _load_plugin(self, plugin_path: Path):
        """Load a single plugin"""
        config_file = plugin_path / "plugin.yaml"
        if not config_file.exists():
            logger.warning(f"No plugin.yaml found in {plugin_path}")
            return
            
        # Load plugin configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        plugin_name = config.get('name', plugin_path.name)
        
        # Check if plugin is enabled
        if not config.get('enabled', True):
            logger.info(f"Plugin {plugin_name} is disabled")
            return
        
        # Load Python module
        module_name = f"{plugin_path.name}_plugin"
        init_file = plugin_path / "__init__.py"
        
        if not init_file.exists():
            logger.warning(f"No __init__.py found in {plugin_path}")
            return
        
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, init_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Get plugin class
        plugin_class_name = config.get('class', 'Plugin')
        if hasattr(module, plugin_class_name):
            plugin_class = getattr(module, plugin_class_name)
            
            # Instantiate plugin
            plugin_instance = plugin_class(config)
            
            # Store plugin
            self.plugins[plugin_name] = {
                'instance': plugin_instance,
                'config': config,
                'module': module,
                'path': plugin_path
            }
            
            logger.info(f"Loaded plugin: {plugin_name} v{config.get('version', 'unknown')}")
        else:
            logger.error(f"Plugin class {plugin_class_name} not found in {module_name}")
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """Get a specific plugin instance"""
        plugin_data = self.plugins.get(name)
        return plugin_data['instance'] if plugin_data else None
    
    def get_all_plugins(self) -> Dict[str, Any]:
        """Get all loaded plugins"""
        return {name: data['instance'] for name, data in self.plugins.items()}
    
    def reload_plugins(self):
        """Reload all plugins"""
        logger.info("Reloading all plugins...")
        
        # Store current plugins
        old_plugins = dict(self.plugins)
        
        try:
            # Reload all plugins
            self.load_plugins()
            
            # Cleanup old plugins
            for name, data in old_plugins.items():
                if hasattr(data['instance'], 'cleanup'):
                    try:
                        data['instance'].cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up plugin {name}: {e}")
            
            logger.info("Plugins reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload plugins: {e}")
            # Restore old plugins on failure
            self.plugins = old_plugins
            raise
    
    def _setup_hot_reload(self):
        """Setup file system watcher for hot reload"""
        try:
            from watchdog.observers import Observer
            
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            self.observer = Observer()
            event_handler = PluginReloadHandler(self)
            self.observer.schedule(event_handler, self.plugin_dir, recursive=True)
            self.observer.start()
            
            logger.info("Hot reload enabled for plugins")
        except ImportError:
            logger.warning("Watchdog not installed, hot reload disabled")
    
    def stop_hot_reload(self):
        """Stop hot reload observer"""
        if self.observer:
            self.observer.stop()
            self.observer.join()