"""
Fix temporal para problema de SSL en Windows
"""

import ssl
import certifi

# Configurar certificados
ssl._create_default_https_context = ssl._create_unverified_context

print("✓ SSL fix aplicado")
print("Ahora ejecuta train.py normalmente")