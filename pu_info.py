import subprocess
import tensorflow as tf

class PUInfo:
    @staticmethod
    def extract_numbers(text):
        '''
        Extracts all the digits from a given text and concatenates them into a single string.
        Args:
            text (str): The input text from which digits will be extracted.
        Returns:
            str: A string containing only the digits found in the input text.
                 Returns an empty string if no digits are found.
        '''
        numbers = ''
        for char in text:
            if char.isdigit():
                numbers += char
        return numbers

    @staticmethod
    def find_cudnn_version_file():
        '''
        Finds the cudnn_version.h file in the /usr/ directory.
        Returns:
            str: The path to the cudnn_version.h file if found, otherwise an empty string.
        '''
        try:
            # 'find /usr/ -name cudnn_version.h'
            locations = subprocess.run(['find', '/usr/', '-name', 'cudnn_version.h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return locations.stdout
        except:
            return ''

    @staticmethod
    def get_cudnn_version():
        '''
        Retrieves the CuDNN version.
        Returns:
            str: The CuDNN version if found, '' if it's not.
        '''
        try:
            location = PUInfo.find_cudnn_version_file().split('\n')[0]
            cudnn_version = ''
            result = subprocess.run(['cat', location], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout

            for line in output.splitlines():
                if 'CUDNN_MAJOR' in line or 'CUDNN_MINOR' in line or 'CUDNN_PATCHLEVEL' in line:
                    cudnn_version += line[-1]
            
            if len(cudnn_version) > 2:
                cudnn_version = '.'.join(PUInfo.extract_numbers(cudnn_version))
            
            return cudnn_version

        except FileNotFoundError:
            print('CuDNN is not installed')
        except subprocess.CalledProcessError:
            print('Error executing subprocess command')
        except TypeError:
            print('Unexpected type error, result.stdout is probably None')
        except Exception:
            print('Unexpected error')

        return ''

    @staticmethod
    def get_nvidia_driver_info():
        '''
        This function attempts to retrieve the NVIDIA driver version using the nvidia-smi command.
        Returns:
            nvidia_info: The NVIDIA Driverr info if found, otherwise an empty string.
        '''
        try:
            # Execute nvidia-smi to get GPU information
            result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, check=True)
            output = result.stdout.decode('utf-8')
            
            nvidia_info = ''
            for line in output.splitlines():
                if 'N/A' not in line:
                    nvidia_info += line + '\n'

            return nvidia_info
            
        except subprocess.CalledProcessError:
            print('nvidia-smi command might not be available')

            return ''

    @staticmethod
    def get_cuda_info():
        '''
        This function attempts to retrieve the CUDA Toolkit version using the nvcc command.
        Returns:
            cuda_info: The CUDA Toolkit info if found, otherwise an empty string.
        '''
        try:
            # Execute nvcc --version to get compiler information
            result = subprocess.run(['nvcc', '--version'], capture_output=True, check=True)
            output = result.stdout.decode('utf-8')

            cuda_info = ''
            for line in output.splitlines():
                cuda_info += line + '\n'
            else:
                return cuda_info
            
        except subprocess.CalledProcessError:
            print('nvcc command might not be available')
        except FileNotFoundError:
            print('nvcc command might not be available')

        return ''

    @staticmethod
    def log_gpu_info():
        '''
        Print information about the available GPUs and CUDA configuration.
        Returns:
            str: with GPU informations.
        '''
        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        if physical_devices:
            num_gpus = len(physical_devices)
            cuda_compiler_flags = tf.sysconfig.get_compile_flags()
            cuda_build_info = tf.sysconfig.get_build_info()
            gpus_details = []
            for gpu_device in physical_devices:
                gpus_details.append(tf.config.experimental.get_device_details(gpu_device))

            gpu_info = ''
            for gpu_details in gpus_details:
                for key, value in gpu_details.items():
                    gpu_info += f'{key.capitalize().replace("_"," ")}: {value}\n'

            cuda_flags_info = '\n'.join([flag for flag in cuda_compiler_flags])

            cuda_build_info_text = ''
            for key, value in cuda_build_info.items():
                cuda_build_info_text += f'{key.capitalize().replace("_"," ")}: {value}\n'
            
            driver_info = f'\n{"="*10}NVIDIA Driver Informations{"="*10}     {PUInfo.get_nvidia_driver_info()}\n' if PUInfo.get_nvidia_driver_info() else ''
            cuda_info = f'{"="*10}CUDA Toolkit Runtime API Version{"="*10}\n{PUInfo.get_cuda_info()}\n' if PUInfo.get_cuda_info() else ''
            cudnn_info = f'{"="*10}cuDNN Version{"="*10}\n{PUInfo.get_cudnn_version()}\n'if PUInfo.get_cudnn_version() else ''

            additional_info = (
                f'TensorFlow Version: {tf.__version__}\n'
            )

            box_content = (
                f'{"="*10}GENERAL INFO{"="*10}\n'
                f'Number of GPUs: {num_gpus}\n'
                f'Physical devices: {physical_devices}\n'
                f'{gpu_info}\n'
                f'{"="*10}CUDA Compiler Flags{"="*10}\n{cuda_flags_info}\n\n'
                f'{"="*10}CUDA Build Information{"="*10}\n{cuda_build_info_text}\n'
                f'{cuda_info}'
                f'{cudnn_info}'
                f'{driver_info}'
                f'Additional Information:\n{additional_info}'
            )

            return box_content
        else:
            raise RuntimeError('GPU not available. Please ensure that your system has a compatible GPU and the necessary drivers installed.')

    @staticmethod
    def all_cpu_info():
        '''
        Retrieve detailed information about the CPU from the /proc/cpuinfo file on Linux systems.
        Returns:
            dict: A dictionary containing CPU information.
        '''
        cpu_info = {}
        with open('/proc/cpuinfo') as f:
            lines = f.readlines()

        for line in lines:
            if ':' in line:
                parts = line.split(':')
                key = parts[0].strip()
                value = parts[1].strip()
                cpu_info[key] = value

        return cpu_info

    @staticmethod
    def log_cpu_info():
        '''
        Log CPU information retrieved from the /proc/cpuinfo file.
        Returns:
            str: A string containing CPU information.
        '''
        info = f'{"="*10}CPU INFO{"="*10}\n'
        cpu_info = PUInfo.all_cpu_info()
        for key, value in cpu_info.items():
            info += f'{key.capitalize()}: {value}\n'
        
        return info
