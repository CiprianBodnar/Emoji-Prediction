from lxml import etree
from io import StringIO
import sys

if __name__ == "__main__":
    dtd_file = sys.argv[1]
    xsd_file = sys.argv[2]
    xml_file = sys.argv[3]

    with open(dtd_file) as dtd, open(xsd_file) as xsd, open(xml_file) as xml:
        dtd_content = dtd.read()
        xsd_content = xsd.read()
        xml_content = xml.read()

        dtd_tree = etree.DTD(StringIO(dtd_content))
        xsd_tree = etree.XMLSchema(etree.fromstring(xsd_content.encode()))
        xml_root = etree.fromstring(xml_content.encode())

        is_dtd_valid = dtd_tree.validate(xml_root)
        is_xsd_valid = xsd_tree.validate(xml_root)

        print("DTD: {}\nXSD: {}".format(is_dtd_valid, is_xsd_valid))
