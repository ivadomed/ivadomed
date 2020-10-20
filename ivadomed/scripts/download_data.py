import os
import shutil
import logging
import cgi
import tempfile
import urllib.parse
import tarfile
import zipfile
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
import sys
import argparse
import textwrap

from ivadomed.utils import init_ivadomed


DICT_URL = {
    "data_example_spinegeneric": {
        "url": ["https://github.com/ivadomed/data_example_spinegeneric/archive/r20200825.zip"],
        "description": "10 randomly picked subject from "
                       "`Spine Generic <https://github.com/spine-generic/data-multi-subject>`_. "
                       "Used for Tutorial and example in Ivadomed."},
    "data_testing": {"url": ["https://github.com/ivadomed/data-testing/archive/r20201020.zip"],
                     "description": "Data Used for integration/unit test in Ivadomed."},
    "t2_tumor": {"url": ["https://github.com/ivadomed/t2_tumor/archive/r20200621.zip"],
                 "description": "Cord tumor segmentation model, trained on T2-weighted contrast."},
    "t2star_sc": {"url": ["https://github.com/ivadomed/t2star_sc/archive/r20200622.zip"],
                  "description": "spinal cord segmentation model, trained on T2-star contrast."},
    "mice_uqueensland_gm": {"url": ["https://github.com/ivadomed/mice_uqueensland_gm/archive/r20200622.zip"],
                            "description": "Gray matter segmentation model on "
                                           "mouse MRI. Data from University of Queensland."},
    "mice_uqueensland_sc": {"url": ["https://github.com/ivadomed/mice_uqueensland_sc/archive/r20200622.zip"],
                            "description": "Cord segmentation model on mouse MRI. Data from University of Queensland."},
    "findcord_tumor": {"url": ["https://github.com/ivadomed/findcord_tumor/archive/r20200621.zip"],
                       "description": "Cord localisation model, trained on T2-weighted images with tumor."},
    "model_find_disc_t1": {"url": ["https://github.com/ivadomed/model_find_disc_t1/archive/r20201013.zip"],
                           "description": "Intervertebral disc detection model trained on T1-weighted images."},
    "model_find_disc_t2": {"url": ["https://github.com/ivadomed/model_find_disc_t2/archive/r20200928.zip"],
                           "description": "Intervertebral disc detection model trained on T2-weighted images."}

}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True,
                        choices=(sorted(DICT_URL)),
                        help="Data to download")
    parser.add_argument("-k", "--keep", required=False, default=False,
                        help="Keep existing data in destination directory")
    parser.add_argument("-o", "--output", required=False,
                        help="Output Folder.")
    return parser


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def download_data(urls):
    """
    Function used to download the data form github or other mirrors
    Args:
        urls (list): List of urls to try.

    Returns:
        downloaded folder path
    """
    if isinstance(urls, str):
        urls = [urls]

    exceptions = []
    # loop through URLs
    for url in urls:
        try:
            logger.info('Trying URL: %s' % url)
            retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
            session = requests.Session()
            session.mount('https://', HTTPAdapter(max_retries=retry))
            response = session.get(url, stream=True)
            response.raise_for_status()

            filename = os.path.basename(urllib.parse.urlparse(url).path)
            if "Content-Disposition" in response.headers:
                _, content = cgi.parse_header(response.headers['Content-Disposition'])
                filename = content["filename"]

            # protect against directory traversal
            filename = os.path.basename(filename)
            if not filename:
                # this handles cases where you're loading something like an index page
                # instead of a specific file. e.g. https://osf.io/ugscu/?action=view.
                raise ValueError("Unable to determine target filename for URL: %s" % (url,))

            tmp_path = os.path.join(tempfile.mkdtemp(), filename)

            logger.info('Downloading: %s' % filename)

            with open(tmp_path, 'wb') as tmp_file:
                total = int(response.headers.get('content-length', 1))

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)

            return tmp_path

        except Exception as e:
            logger.warning("Link download error, trying next mirror (error was: %s)" % e)
            exceptions.append(e)
    else:
        raise Exception('Download error', exceptions)


def unzip(compressed, dest_folder):
    """
    Extract compressed file to the dest_folder. Can handle .zip, .tar.gz.
    """
    logger.info('Unzip data to: %s' % dest_folder)

    formats = {'.zip': zipfile.ZipFile,
               '.tar.gz': tarfile.open,
               '.tgz': tarfile.open}
    for format, open in formats.items():
        if compressed.lower().endswith(format):
            break
    else:
        raise TypeError('ERROR: The file %s is of wrong format' % (compressed,))

    try:
        open(compressed).extractall(dest_folder)
    except:
        print('ERROR: ZIP package corrupted. Please try downloading again.')
        raise


def _format_bundles():
    def format_bundle(name, values):
        return f'`{name} <{values["url"]}>`_ : {values["description"]}'
    return str.join("\n", ["* %s" % format_bundle(name, values) for name, values in DICT_URL.items()])

def install_data(url, dest_folder, keep=False):
    """
    Download a data bundle from an URL and install it in the destination folder.

    Usage example ::

        ivadomed_download_data -d data_testing -o ivado_testing_data


    Existing data bundles:

{BUNDLES}

    .. note::
        The function tries to be smart about the data contents.
        Examples:


        a. If the archive only contains a `README.md`, and the destination folder is `${{dst}}`,
        `${{dst}}/README.md` will be created.
        Note: an archive not containing a single folder is commonly known as a "bomb" because
        it puts files anywhere in the current working directory.( see `Tarbomb
        <https://en.wikipedia.org/wiki/Tar_(computing)#Tarbomb>`_)


        b. If the archive contains a `${{dir}}/README.md`, and the destination folder is `${{dst}}`,
        `${{dst}}/README.md` will be created.
        Note: typically the package will be called `${{basename}}-${{revision}}.zip` and contain
        a root folder named `${{basename}}-${{revision}}/` under which all the other files will
        be located.
        The right thing to do in this case is to take the files from there and install them
        in `${{dst}}`.
        - Uses `download_data()` to retrieve the data.
        - Uses `unzip()` to extract the bundle.

    Args:
        url (string): URL or sequence thereof (if mirrors). For this package there is a dictionnary
            listing existing data bundle with their url. Type ivadomed_download_data -h to see possible value. Flag ``-d``
        dest_folder (string): destination directory for the data (to be created). If not used the output folder
            will be the name of the data bundle. Flag ``-o``, ``--output``
        keep (bool): whether to keep existing data in the destination folder (if it exists). Flag ``-k``, ``--keep``
    """

    if not keep and os.path.exists(dest_folder):
        logger.warning("Removing existing destination folder “%s”", dest_folder)
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)

    tmp_file = download_data(url)

    extraction_folder = tempfile.mkdtemp()

    unzip(tmp_file, extraction_folder)

    # Identify whether we have a proper archive or a tarbomb
    with os.scandir(extraction_folder) as it:
        has_dir = False
        nb_entries = 0
        for entry in it:
            if entry.name in ("__MACOSX",):
                continue
            nb_entries += 1
            if entry.is_dir():
                has_dir = True

    if nb_entries == 1 and has_dir:
        # tarball with single-directory -> go under
        with os.scandir(extraction_folder) as it:
            for entry in it:
                if entry.name in ("__MACOSX",):
                    continue
                bundle_folder = entry.path
    else:
        # bomb scenario -> stay here
        bundle_folder = extraction_folder

    # Copy over
    for cwd, ds, fs in os.walk(bundle_folder):
        ds.sort()
        fs.sort()
        ds[:] = [d for d in ds if d not in ("__MACOSX",)]
        for d in ds:
            srcpath = os.path.join(cwd, d)
            relpath = os.path.relpath(srcpath, bundle_folder)
            dstpath = os.path.join(dest_folder, relpath)
            if os.path.exists(dstpath):
                # lazy -- we assume existing is a directory, otherwise it will crash safely
                logger.debug("- d- %s", relpath)
            else:
                logger.debug("- d+ %s", relpath)
                os.makedirs(dstpath)

        for f in fs:
            srcpath = os.path.join(cwd, f)
            relpath = os.path.relpath(srcpath, bundle_folder)
            dstpath = os.path.join(dest_folder, relpath)
            if os.path.exists(dstpath):
                logger.debug("- f! %s", relpath)
                logger.warning("Updating existing “%s”", dstpath)
                os.unlink(dstpath)
            else:
                logger.debug("- f+ %s", relpath)
            shutil.copy(srcpath, dstpath)

    logger.info("Removing temporary folders...")
    shutil.rmtree(os.path.dirname(tmp_file))
    shutil.rmtree(extraction_folder)


# This line allows to format the `install_data()` docstrings, because this formatting
# cannot be done in the function directly. 
# `create_string()` is a custom function that converts our dict into a string
# which is easier to add in the documentation.
install_data.__doc__=install_data.__doc__.format(BUNDLES=textwrap.indent(_format_bundles(), ' '*6))


def main(args=None):
    init_ivadomed()

    # Dictionary containing list of URLs for data names.
    # Mirror servers are listed in order of decreasing priority.
    # If exists, favour release artifact straight from github

    if args is None:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse_args()
    data_name = arguments.d
    if arguments.output is None:
        dest_folder = os.path.join(os.path.abspath(os.curdir), data_name)
    else:
        dest_folder = arguments.output

    url = DICT_URL[data_name]["url"]
    install_data(url, dest_folder, keep=arguments.keep)
    return 0


if __name__ == '__main__':
    main()
