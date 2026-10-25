import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ('build.sh', 'install.sh', 'develop.sh')


class TestBuildScripts(unittest.TestCase):
    def run_script(self, script, build_status, install_status=0):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        project, commands, caller = root / 'project', root / 'bin', root / 'caller'
        for directory in (project / 'deep_ep', commands, caller):
            directory.mkdir(parents=True)
        shutil.copyfile(ROOT / script, project / script)

        # A failed build can leave partial artifacts behind.
        (commands / 'python').write_text(
            '#!/bin/sh\n'
            'printf "python\\n" >> "$DEEPEP_TEST_LOG"\n'
            'mkdir -p build/lib.test/deep_ep dist\n'
            'touch build/lib.test/deep_ep/_C.so dist/deep_ep-test.whl\n'
            'exit "$DEEPEP_TEST_BUILD_STATUS"\n'
        )
        (commands / 'pip').write_text(
            '#!/bin/sh\n'
            'printf "pip\\n" >> "$DEEPEP_TEST_LOG"\n'
            'exit "$DEEPEP_TEST_INSTALL_STATUS"\n'
        )
        for executable in commands.iterdir():
            executable.chmod(0o755)

        log = root / 'commands.log'
        env = os.environ | {
            'PATH': f'{commands}{os.pathsep}{os.environ["PATH"]}',
            'DEEPEP_TEST_LOG': str(log),
            'DEEPEP_TEST_BUILD_STATUS': str(build_status),
            'DEEPEP_TEST_INSTALL_STATUS': str(install_status),
        }
        result = subprocess.run(
            ['bash', str(project / script)], cwd=caller, env=env,
            capture_output=True, text=True, timeout=10,
        )
        return result, log.read_text().splitlines(), project

    def test_failed_build_stops_the_script(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, commands, project = self.run_script(script, 37)
                self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
                self.assertEqual(commands, ['python'])
                self.assertFalse((project / 'deep_ep' / '_C.so').is_symlink())

    def test_failed_install_is_reported(self):
        result, commands, _ = self.run_script('install.sh', 0, install_status=41)
        self.assertEqual(result.returncode, 41, result.stdout + result.stderr)
        self.assertEqual(commands, ['python', 'pip'])

    def test_successful_build_completes(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, commands, project = self.run_script(script, 0)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(commands, ['python', 'pip'] if script == 'install.sh' else ['python'])
                if script == 'develop.sh':
                    self.assertTrue((project / 'deep_ep' / '_C.so').is_symlink())
                    self.assertTrue((project / 'deep_ep' / '_C.so').exists())


if __name__ == '__main__':
    unittest.main()
