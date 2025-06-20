RELEASE_TYPE: patch

<<<<<<< HEAD
Fixes a substantial performance regression in stateful tests from computing string representations, present since :ref:`version 6.131.20 <v6.131.20>`.
||||||| ed6e36c2f
=======
Speed up usages of |st.sampled_from| by deferring evaluation of its repr, and truncating its repr for large collections (over 512 elements). This is especially noticeable when using |st.sampled_from| with large collections. The repr of |st.sampled_from| strategies involving sequence classes with custom reprs may change as a result of this release.
>>>>>>> sampledfrom-repr-speed
