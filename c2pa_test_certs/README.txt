These are the public test signing certificate and key shipped with c2pa-python
(tests/fixtures/es256_certs.pem and es256_private.key, https://github.com/contentauth/c2pa-python),
dual licensed under Apache-2.0 or MIT. They are for building the stage 3 test set only.
Anything signed with them validates as "Valid" but reports signingCredential.untrusted,
because the certificate is not on the C2PA trust list. Never use them to sign real content.
