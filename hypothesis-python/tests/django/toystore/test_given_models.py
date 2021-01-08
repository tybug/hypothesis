# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# CONTRIBUTING.rst for a full list of people who may hold copyright, and
# consult the git log if you need to determine who owns an individual
# contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# END HEADER

import datetime as dt
from uuid import UUID

from django.conf import settings as django_settings
from django.contrib.auth.models import User

from hypothesis import HealthCheck, assume, given, infer, settings
from hypothesis.control import reject
from hypothesis.errors import HypothesisException, InvalidArgument
from hypothesis.extra.django import (
    TestCase,
    TransactionTestCase,
    from_model,
    register_field_strategy,
)
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.strategies import binary, just, lists
from tests.django.toystore.models import (
    Car,
    Company,
    CompanyExtension,
    CouldBeCharming,
    Customer,
    Customish,
    CustomishDefault,
    CustomishField,
    MandatoryComputed,
    ManyNumerics,
    ManyTimes,
    OddFields,
    RestrictedFields,
    SelfLoop,
    Store,
)

register_field_strategy(CustomishField, just("a"))


class TestGetsBasicModels(TestCase):
    @given(from_model(Company))
    def test_is_company(self, company):
        self.assertIsInstance(company, Company)
        self.assertIsNotNone(company.pk)

    @given(from_model(Store, company=from_model(Company)))
    def test_can_get_a_store(self, store):
        assert store.company.pk

    @given(lists(from_model(Company)))
    def test_can_get_multiple_models_with_unique_field(self, companies):
        assume(len(companies) > 1)
        for c in companies:
            self.assertIsNotNone(c.pk)
        self.assertEqual(
            len({c.pk for c in companies}), len({c.name for c in companies})
        )

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(from_model(Customer))
    def test_is_customer(self, customer):
        self.assertIsInstance(customer, Customer)
        self.assertIsNotNone(customer.pk)
        self.assertIsNotNone(customer.email)

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(from_model(Customer))
    def test_tz_presence(self, customer):
        if django_settings.USE_TZ:
            self.assertIsNotNone(customer.birthday.tzinfo)
        else:
            self.assertIsNone(customer.birthday.tzinfo)

    @given(from_model(CouldBeCharming))
    def test_is_not_charming(self, not_charming):
        self.assertIsInstance(not_charming, CouldBeCharming)
        self.assertIsNotNone(not_charming.pk)
        self.assertIsNone(not_charming.charm)

    @given(from_model(SelfLoop))
    def test_sl(self, sl):
        self.assertIsNone(sl.me)

    @given(lists(from_model(ManyNumerics)))
    def test_no_overflow_in_integer(self, manyints):
        pass

    @given(from_model(Customish))
    def test_custom_field(self, x):
        assert x.customish == "a"

    def test_mandatory_fields_are_mandatory(self):
        self.assertRaises(InvalidArgument, from_model(Store).example)

    def test_mandatory_computed_fields_are_mandatory(self):
        with self.assertRaises(InvalidArgument):
            from_model(MandatoryComputed).example()

    def test_mandatory_computed_fields_may_not_be_provided(self):
        mc = from_model(MandatoryComputed, company=from_model(Company))
        self.assertRaises(RuntimeError, mc.example)

    @given(from_model(CustomishDefault, customish=infer))
    def test_customish_default_overridden_by_infer(self, x):
        assert x.customish == "a"

    @given(from_model(CustomishDefault, customish=infer))
    def test_customish_infer_uses_registered_instead_of_default(self, x):
        assert x.customish == "a"

    @given(from_model(OddFields))
    def test_odd_fields(self, x):
        assert isinstance(x.uuid, UUID)
        assert isinstance(x.slug, str)
        assert " " not in x.slug
        assert isinstance(x.ipv4, str)
        assert len(x.ipv4.split(".")) == 4
        assert all(int(i) in range(256) for i in x.ipv4.split("."))
        assert isinstance(x.ipv6, str)
        assert set(x.ipv6).issubset(set("0123456789abcdefABCDEF:."))

    @given(from_model(ManyTimes))
    def test_time_fields(self, x):
        assert isinstance(x.time, dt.time)
        assert isinstance(x.date, dt.date)
        assert isinstance(x.duration, dt.timedelta)

    @given(from_model(Company))
    def test_no_null_in_charfield(self, x):
        # regression test for #1045.  Company just has a convenient CharField.
        assert "\x00" not in x.name

    @given(binary(min_size=10))
    def test_foreign_key_primary(self, buf):
        # Regression test for #1307
        company_strategy = from_model(Company, name=just("test"))
        strategy = from_model(
            CompanyExtension, company=company_strategy, self_modifying=just(2)
        )
        try:
            ConjectureData.for_buffer(buf).draw(strategy)
        except HypothesisException:
            reject()
        # Draw again with the same buffer. This will cause a duplicate
        # primary key.
        ConjectureData.for_buffer(buf).draw(strategy)
        assert CompanyExtension.objects.all().count() == 1


class TestsNeedingRollback(TransactionTestCase):
    def test_can_get_examples(self):
        for _ in range(200):
            from_model(Company).example()


class TestRestrictedFields(TestCase):
    @given(from_model(RestrictedFields))
    def test_constructs_valid_instance(self, instance):
        self.assertIsInstance(instance, RestrictedFields)
        instance.full_clean()
        self.assertLessEqual(len(instance.text_field_4), 4)
        self.assertLessEqual(len(instance.char_field_4), 4)
        self.assertIn(instance.choice_field_text, ("foo", "bar"))
        self.assertIn(instance.choice_field_int, (1, 2))
        self.assertIn(instance.null_choice_field_int, (1, 2, None))
        self.assertEqual(
            instance.choice_field_grouped, instance.choice_field_grouped.lower()
        )
        self.assertEqual(instance.even_number_field % 2, 0)
        self.assertTrue(instance.non_blank_text_field)


class TestValidatorInference(TestCase):
    @given(from_model(User))
    def test_user_issue_1112_regression(self, user):
        assert user.username


class TestPosOnlyArg(TestCase):
    @given(from_model(Car))
    def test_user_issue_2369_regression(self, val):
        pass

    def test_from_model_argspec(self):
        self.assertRaises(TypeError, from_model().example)
        self.assertRaises(TypeError, from_model(Car, None).example)
        self.assertRaises(TypeError, from_model(model=Customer).example)
