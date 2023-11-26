// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Linear__h__
#define __ivqML__Model__Linear__h__

#include <ivqML/Model/Base.h>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _S >
    class Linear
      : public ivqML::Model::Base< _S >
    {
    public:
      using Self = Linear;
      using Superclass = ivqML::Model::Base< _S >;

      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;

    public:
      Linear( const TNatural& n = 0 );
      virtual ~Linear( ) override = default;

      virtual void set_number_of_parameters( const TNatural& p ) override;

      virtual TNatural number_of_inputs( ) const override;
      virtual void set_number_of_inputs( const TNatural& p ) override;

      virtual TNatural number_of_outputs( ) const override;

    protected:
      virtual TNatural _cache_size( ) const override;
      virtual void _resize_cache( const TNatural& s ) const override;
      virtual TMap& _input_cache( ) const override;
      virtual const TMap& _output_cache( ) const override;
    
      virtual void _evaluate( const TNatural& m ) const override;
      // TODO: virtual void _cost( TScalar* J ) = 0;

    protected:
      mutable TMap m_T { nullptr, 0, 0 };
      mutable TMap m_Xi { nullptr, 0, 0 };
      mutable TMap m_X { nullptr, 0, 0 };
      mutable TMap m_Y { nullptr, 0, 0 };
    };
  } // end namespace
} // end namespace

///// #include <ivqML/Model/Linear.hxx>

#endif // __ivqML__Model__Linear__h__

// eof - $RCSfile$
